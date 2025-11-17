from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.value_head import LatentValueHead
from models.edit_policy import EditPolicyHead
from utils.lipschitz import apply_spectral_norm_to_trm, apply_spectral_norm_to_value_head

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # RL / value-head / contraction dials (defaults keep them OFF)
    rl_enable_value_head: bool = False
    rl_enable_contraction: bool = False
    rl_value_hidden_dim: int = 256
    rl_target_Lz: float = 0.9
    rl_target_Lv: float = 1.0
    rl_enable_policy_head: bool = False
    rl_num_actions: int = 0   # total number of discrete actions; last index is STOP

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def build_seq_info(self) -> Dict[str, Optional[CosSin]]:
        return dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

    def build_latent_context(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            "seq_info": self.build_seq_info(),
            "input_embeddings": self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]),
        }

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None):
        device = device or self.H_init.device
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        reset_flag = reset_flag.to(self.H_init.device)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def latent_step(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        z_H, z_L = carry.z_H, carry.z_L
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        latent_context = self.build_latent_context(batch)
        input_embeddings = latent_context["input_embeddings"]
        seq_info = latent_context["seq_info"]

        z = carry
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                z = self.latent_step(z, input_embeddings, seq_info)
        # 1 with grad
        z = self.latent_step(z, input_embeddings, seq_info)
        z_H, z_L = z.z_H, z.z_L

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)
        self.z_dim = self.config.hidden_size
        self.x_embed_dim = self.config.hidden_size
        self.y_embed_dim = self.config.hidden_size  # placeholder pooled representation of y

        if self.config.rl_enable_value_head:
            self.value_head = LatentValueHead(
                z_dim=self.z_dim,
                x_dim=self.x_embed_dim,
                hidden_dim=self.config.rl_value_hidden_dim,
            )
        else:
            self.value_head = None

        if self.config.rl_enable_policy_head:
            if self.config.rl_num_actions <= 0:
                raise ValueError("rl_num_actions must be > 0 when rl_enable_policy_head=True")
            self.edit_policy = EditPolicyHead(
                latent_dim=self.z_dim,
                x_embed_dim=self.x_embed_dim,
                y_embed_dim=self.y_embed_dim,
                action_dim=self.config.rl_num_actions,
            )
        else:
            self.edit_policy = None

        if self.config.rl_enable_contraction:
            apply_spectral_norm_to_trm(self.inner)
            if self.value_head is not None:
                apply_spectral_norm_to_value_head(self.value_head)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def _standardize_latent_batch(self, x: Any, y: Any) -> Dict[str, torch.Tensor]:
        """
        Normalize the latent helper inputs to the batch dict format expected by the inner model.
        """
        if not isinstance(x, dict):
            raise TypeError("Expected `x` to be a batch dict containing `inputs` and `puzzle_identifiers`.")
        if "inputs" not in x or "puzzle_identifiers" not in x:
            missing = {"inputs", "puzzle_identifiers"} - set(x.keys())
            raise KeyError(f"Missing required keys for latent helpers: {missing}.")
        return x

    def _resolve_latent_context(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Returns cached (or freshly computed) latent context for update_latent-style calls.
        """
        context = batch.get("_latent_context")
        if context is None:
            context = self.inner.build_latent_context(batch)
        return context

    def init_latent(self, x: Any, y: Any) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        Initialize z^(0) from (x, y) by reusing the standard inner carry reset logic.
        """
        batch = self._standardize_latent_batch(x, y)
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        empty_carry = self.inner.empty_carry(batch_size, device=device)
        reset_flag = torch.ones(batch_size, dtype=torch.bool, device=device)
        return self.inner.reset_carry(reset_flag, empty_carry)

    def update_latent(
        self,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        y: Any,
        x: Any,
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        One application of the inner map f_theta(z, y, x) -> z_next.
        """
        batch = self._standardize_latent_batch(x, y)
        context = self._resolve_latent_context(batch)
        return self.inner.latent_step(z, context["input_embeddings"], context["seq_info"])

    def unroll_latent(
        self,
        x: Any,
        y: Any,
        n: int,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, List[TinyRecursiveReasoningModel_ACTV1InnerCarry]]:
        """
        Run the inner recursion for n steps starting from z^(0).
        """
        batch = dict(self._standardize_latent_batch(x, y))
        batch["_latent_context"] = self.inner.build_latent_context(batch)

        z = self.init_latent(batch, y)
        zs: List[TinyRecursiveReasoningModel_ACTV1InnerCarry] = [z]
        for _ in range(n):
            z = self.update_latent(z, y, batch)
            zs.append(z)
        return z, zs

    def eval_latent(self, x: Any, y: Any, n: int) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        Convenience wrapper used by RL code. Runs unroll_latent and returns only z^(n).
        """
        z_n, _ = self.unroll_latent(x, y, n)
        return z_n

    def used_value(self, x: Dict[str, torch.Tensor], y: Any, n: int) -> torch.Tensor:
        """
        Compute U_n(s) = V_ψ(z^(n)(s), x) for a batch.

        - x: batch dict with at least ["inputs", "puzzle_identifiers"]
        - y: plan (currently unused, kept for future RL integration)
        - n: number of inner latent steps
        """
        if self.value_head is None:
            raise RuntimeError("Value head is not enabled; set rl_enable_value_head=True in the config.")

        # 1) Run episodic latent unrolling to get z^(n)
        z_n, _ = self.unroll_latent(x, y, n)
        z_vec = z_n.z_H.mean(dim=1)

        # 2) Build a fresh latent context to obtain input embeddings and summarize x
        batch = self._standardize_latent_batch(x, y)
        latent_context = self.inner.build_latent_context(batch)
        input_embeddings = latent_context["input_embeddings"]
        x_embed = input_embeddings.mean(dim=1)

        # 3) Apply value head
        return self.value_head(z_vec, x_embed)

    def policy_dist(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """
        Produce a Categorical distribution over edit actions given (x, y).

        - x: batch dict with at least ["inputs", "puzzle_identifiers"]
        - y: plan tensor (same batch size), currently assumed to have shape compatible with inputs.
        - n: number of inner latent steps for the evaluator.
        - action_mask: optional [B, action_dim] boolean mask.
        """

        if self.edit_policy is None:
            raise RuntimeError(
                "Policy head is not enabled; set rl_enable_policy_head=True and rl_num_actions>0 in the config."
            )

        # 1) Get z^(n) via episodic latent unrolling
        z_n, _ = self.unroll_latent(x, y, n)
        z_vec = z_n.z_H.mean(dim=1)

        # 2) Summarize x via latent context embeddings
        batch = self._standardize_latent_batch(x, y)
        latent_context = self.inner.build_latent_context(batch)
        input_embeddings = latent_context["input_embeddings"]
        x_embed = input_embeddings.mean(dim=1)

        # 3) For now, we treat y embedding as the same as x embedding (placeholder).
        #    Later this can be replaced with a true plan encoder.
        y_embed = x_embed

        return self.edit_policy(z_vec, x_embed, y_embed, action_mask=action_mask)

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
