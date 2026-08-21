from typing import Tuple, List, Dict, Optional, Any, Literal
from dataclasses import dataclass
import math
import warnings
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from pydantic import BaseModel, model_validator

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.value_head import LatentValueHead
from models.edit_policy import EditPolicyHead
from utils.lipschitz import (
    apply_spectral_norm_to_trm,
    apply_spectral_norm_to_value_head,
    enforce_global_contraction,
    enforce_global_contraction_on_value_head,
    apply_opnorm_clamp_to_trm,
)
from utils.compute_accounting import ModelComputeCounters, validate_model_counters

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

    # ==========================================================================
    # RL-SPECIFIC CONFIGURATION (UPI-TRM Extension)
    # ==========================================================================
    # 
    # IMPORTANT: These flags should be LEFT AS FALSE for vanilla TRM experiments
    # (supervised pretraining). They are ONLY meant for the UPI-TRM RL integration.
    # 
    # When any rl_enable_* flag is True, the model gains extra parameters
    # (value head, policy head, etc.) that will be trained during RL but
    # should NOT be present during vanilla supervised TRM pretraining.
    # 
    # For vanilla TRM (supervised):
    #   - rl_enable_value_head = False
    #   - rl_enable_policy_head = False  
    #   - rl_enable_contraction = False
    #   - rl_enable_z_init_encoder = False
    # 
    # For UPI-TRM (RL):
    #   - rl_enable_value_head = True (adds LatentValueHead for V_ψ)
    #   - rl_enable_policy_head = True (adds EditPolicyHead for π_θ)
    #   - rl_enable_contraction = True (optional contraction-oriented intervention)
    #   - Set rl_num_actions to the discrete action space size
    # ==========================================================================
    
    rl_enable_value_head: bool = False  # Add V_ψ(z, x) value head
    rl_enable_contraction: bool = False  # Apply spectral norm + contraction scaling
    rl_value_hidden_dim: int = 256  # Hidden dimension of value head MLP
    rl_target_Lz: float = 0.9  # Configured recurrent-modulus target L_z < 1
    rl_target_Lv: float = 1.0  # Target Lipschitz of value head w.r.t. z
    rl_disable_value_head_norm: bool = False  # Skip value head normalization (2x2 ablation finding)
    rl_enable_policy_head: bool = False  # Add edit policy head π_θ(a|s)
    rl_num_actions: int = 0   # Total discrete actions; must be > 0 if policy head enabled
    rl_enable_z_init_encoder: bool = False  # Use (x,y)-dependent z initialization
    
    # === Forward-invariant recurrent projection ===
    # Enabled mode applies Euclidean projection Pi_R with a finite R > 0.
    # Disabled mode applies the identity operator and has no radius.
    rl_latent_projection_mode: Literal["enabled", "disabled"] = "disabled"
    rl_latent_ball_radius: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_projection_config(cls, data: Any) -> Any:
        """Migrate historical radius-only model payloads to an explicit mode."""

        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        if "rl_latent_projection_mode" in migrated:
            if (
                migrated["rl_latent_projection_mode"] == "disabled"
                and "rl_latent_ball_radius" not in migrated
            ):
                migrated["rl_latent_ball_radius"] = None
            return migrated
        if "rl_latent_ball_radius" not in migrated:
            return migrated

        radius = migrated["rl_latent_ball_radius"]
        if radius == 0 or radius == 0.0:
            warnings.warn(
                "Legacy model config used rl_latent_ball_radius=0 to disable "
                "projection; migrate to rl_latent_projection_mode='disabled' "
                "with rl_latent_ball_radius=None.",
                DeprecationWarning,
                stacklevel=3,
            )
            migrated["rl_latent_projection_mode"] = "disabled"
            migrated["rl_latent_ball_radius"] = None
        elif radius is not None:
            warnings.warn(
                "Legacy model config omitted rl_latent_projection_mode; "
                "inferred 'enabled' from its positive projection radius.",
                DeprecationWarning,
                stacklevel=3,
            )
            migrated["rl_latent_projection_mode"] = "enabled"
        return migrated

    @model_validator(mode="after")
    def _validate_projection_config(self) -> "TinyRecursiveReasoningModel_ACTV1Config":
        radius = self.rl_latent_ball_radius
        if self.rl_latent_projection_mode == "enabled":
            if radius is None or not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(
                    "Enabled recurrent projection requires a finite "
                    "rl_latent_ball_radius > 0."
                )
            forward_dtype = getattr(torch, self.forward_dtype, None)
            if not isinstance(forward_dtype, torch.dtype):
                raise ValueError(
                    "Enabled recurrent projection requires a floating-point "
                    "forward_dtype."
                )
            dtype_limits = torch.finfo(forward_dtype)
            if radius < dtype_limits.tiny or radius > dtype_limits.max:
                raise ValueError(
                    "Enabled recurrent projection radius must remain finite and "
                    "positive when represented in forward_dtype; require "
                    f"{dtype_limits.tiny} <= R <= {dtype_limits.max}."
                )
        elif radius is not None:
            raise ValueError(
                "Disabled recurrent projection uses the identity operator and "
                "requires rl_latent_ball_radius=None."
            )
        return self

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
        # Use the carry's device to ensure consistency with input batch
        device = carry.z_H.device
        reset_flag = reset_flag.to(device)
        # H_init/L_init are [hidden_size], need to unsqueeze to [1, 1, hidden_size]
        # for proper broadcasting with carry.z_H/z_L which are [batch_size, seq_len, hidden_size]
        H_init = self.H_init.unsqueeze(0).unsqueeze(0).to(device)
        L_init = self.L_init.unsqueeze(0).unsqueeze(0).to(device)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def latent_step(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        next_carry, _, _ = self.latent_step_with_projection_info(
            carry,
            input_embeddings,
            seq_info,
        )
        return next_carry

    def latent_step_with_projection_info(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> Tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply one step and expose pre-projection norm and activation."""

        pre_projection_carry = self.latent_step_pre_projection(
            carry,
            input_embeddings,
            seq_info,
        )
        z_H, z_L = pre_projection_carry.z_H, pre_projection_carry.z_L
        pre_projection_norm = (
            self._joint_carry_geometry(z_H, z_L)[0].squeeze(-1).squeeze(-1)
        )

        # === Forward-invariant recurrent projection ===
        # Enabled: Euclidean Pi_R for a validated finite R > 0.
        # Disabled: the recurrent projection operator is exactly the identity.
        projection_active = torch.zeros_like(
            pre_projection_norm,
            dtype=torch.bool,
        )
        if self.config.rl_latent_projection_mode == "enabled":
            R = self.config.rl_latent_ball_radius
            assert R is not None
            projection_active = pre_projection_norm > R
            z_H, z_L = self._project_carry_to_ball(z_H, z_L, R)

        return (
            TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L),
            pre_projection_norm,
            projection_active,
        )

    def latent_step_pre_projection(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """Apply the production recurrent update before radial projection."""

        z_H, z_L = carry.z_H, carry.z_L
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H,
            z_L=z_L,
        )
    
    def _project_carry_to_ball(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project the joint recurrent carry under its Euclidean product norm."""

        joint_norm, unit_H, unit_L = self._joint_carry_geometry(z_H, z_L)
        radius_tensor = z_H.new_tensor(radius)
        outside = joint_norm > radius_tensor
        return (
            torch.where(outside, unit_H * radius_tensor, z_H),
            torch.where(outside, unit_L * radius_tensor, z_L),
        )

    @staticmethod
    def _joint_carry_geometry(
        z_H: torch.Tensor,
        z_L: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a stable joint norm and unit direction for one carry batch."""

        max_abs = torch.maximum(
            z_H.abs().amax(dim=(1, 2), keepdim=True),
            z_L.abs().amax(dim=(1, 2), keepdim=True),
        )
        safe_max_abs = torch.where(
            max_abs > 0.0,
            max_abs,
            torch.ones_like(max_abs),
        )
        scaled_H = z_H / safe_max_abs
        scaled_L = z_L / safe_max_abs
        accumulation_dtype = (
            torch.float32
            if z_H.dtype in (torch.float16, torch.bfloat16)
            else z_H.dtype
        )
        scaled_norm = torch.sqrt(
            scaled_H.to(accumulation_dtype).square().sum(
                dim=(1, 2),
                keepdim=True,
            )
            + scaled_L.to(accumulation_dtype).square().sum(
                dim=(1, 2),
                keepdim=True,
            )
        )
        joint_norm = (max_abs.to(accumulation_dtype) * scaled_norm).to(z_H.dtype)
        safe_scaled_norm = torch.where(
            scaled_norm > 0.0,
            scaled_norm,
            torch.ones_like(scaled_norm),
        ).to(z_H.dtype)
        return (
            joint_norm,
            scaled_H / safe_scaled_norm,
            scaled_L / safe_scaled_norm,
        )

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


class ZInitEncoder(nn.Module):
    """
    Encoder network that produces z^(0) from (x, y) embeddings.
    
    This implements the z_init(x, y) function from the paper (Section 4),
    allowing the initial latent state to depend on the current instance and plan.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        # MLP that maps pooled (x, y) embeddings to z^(0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * output_dim),
        )
        
        # Initialize to small values so initial behavior is close to global init
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_embed: [B, input_dim] pooled input embedding
            y_embed: [B, input_dim] pooled plan embedding
        
        Returns:
            z_init: [B, seq_len, output_dim] initial latent state
        """
        combined = torch.cat([x_embed, y_embed], dim=-1)  # [B, input_dim * 2]
        out = self.mlp(combined)  # [B, seq_len * output_dim]
        return out.view(-1, self.seq_len, self.output_dim)


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)
        self._compute_counters = ModelComputeCounters()
        
        seq_len = self.config.seq_len
        hidden_size = self.config.hidden_size
        
        # Value head dimensions: use FLATTENED representations to preserve position info
        # Similar to Policy head, we need spatial structure for Sudoku.
        puzzle_emb_len = self.inner.puzzle_emb_len  # 0 when puzzle_emb_ndim == 0
        total_seq_len = seq_len + puzzle_emb_len
        
        self.z_dim_pooled = hidden_size  # Legacy
        self.x_embed_dim_pooled = hidden_size  # Legacy
        
        # Flattened dimensions
        self.z_dim_flat = total_seq_len * hidden_size
        self.x_embed_dim_flat = total_seq_len * hidden_size
        self.y_embed_dim_flat = total_seq_len * hidden_size
        
        # Combined dimension for Value Head (z + x_embed + y_embed)
        # Note: x and y embeddings are combined before value head
        self.xy_embed_dim_flat = self.x_embed_dim_flat + self.y_embed_dim_flat
        
        # Legacy aliases for compatibility
        self.z_dim = self.z_dim_pooled
        self.x_embed_dim = self.x_embed_dim_pooled
        self.plan_embed_dim = hidden_size
        self.xy_embed_dim = self.x_embed_dim + self.plan_embed_dim

        self.value_head: Optional[LatentValueHead]
        if self.config.rl_enable_value_head:
            self.value_head = LatentValueHead(
                z_dim=self.z_dim_flat,  # Use flattened z
                x_dim=self.xy_embed_dim_flat,  # Use flattened context (x + y)
                hidden_dim=self.config.rl_value_hidden_dim,
            )
        else:
            self.value_head = None

        self.edit_policy: Optional[EditPolicyHead]
        if self.config.rl_enable_policy_head:
            if self.config.rl_num_actions <= 0:
                raise ValueError("rl_num_actions must be > 0 when rl_enable_policy_head=True")
            # BUG FIX: Use flattened dimensions to preserve positional information!
            self.edit_policy = EditPolicyHead(
                latent_dim=self.z_dim_flat,
                x_embed_dim=self.x_embed_dim_flat,
                y_embed_dim=self.y_embed_dim_flat,
                action_dim=self.config.rl_num_actions,
            )
        else:
            self.edit_policy = None

        # Optional (x,y)-dependent initialization encoder
        self.z_init_encoder: Optional[ZInitEncoder]
        if getattr(self.config, "rl_enable_z_init_encoder", False):
            puzzle_emb_len = self.inner.puzzle_emb_len
            total_seq_len = self.config.seq_len + puzzle_emb_len
            self.z_init_encoder = ZInitEncoder(
                input_dim=self.x_embed_dim_pooled,  # Uses pooled embeddings
                hidden_dim=self.config.rl_value_hidden_dim,
                output_dim=self.z_dim_pooled * 2,  # Double output dim to separate H and L init
                seq_len=total_seq_len,
            )
        else:
            self.z_init_encoder = None

        # === Optional contraction-oriented intervention ===
        #
        # WARNING: Enabling rl_enable_contraction is NOT compatible with loading
        # vanilla pretrained TRM weights without fine-tuning!
        #
        # When enabled, this applies operator-norm clamping and output scaling
        # intended to reduce the recurrent modulus. This modifies network behavior:
        # - opnorm clamp: Rescales each layer to ||W|| <= 1 (1-Lipschitz per layer)
        # - Output scaling: Applies a configured recurrent-path scaling factor
        #
        # NOTE: We use opnorm_clamp instead of spectral_norm because finite
        # diagnostics on this architecture recorded local-Lz estimates above
        # 10^5 with spectral_norm in the retired exploratory implementation.
        #
        # If you load a checkpoint that was trained WITHOUT contraction:
        # - The pretrained weights will be rescaled by the contraction factors
        # - The original supervised performance will NOT be preserved
        # - You must fine-tune the model with the new constraints
        #
        # This intervention is not a certificate of the uniform global modulus
        # required by the recurrent-contraction specialization. It only supplies
        # a controlled architectural factor.
        # For practical RL that builds on pretrained TRM, consider:
        # - Training with contraction from scratch, OR
        # - Disabling the intervention to preserve pretrained behavior. The
        #   finite-reference certificate does not require recurrent contraction.
        if self.config.rl_enable_contraction:
            import warnings
            warnings.warn(
                "rl_enable_contraction=True: Applying contraction-oriented "
                "operator-norm clamping and scaling to z->z path layers. This "
                "does not certify the uniform recurrent modulus and "
                "will modify network behavior. "
                "Pretrained weights from vanilla TRM may require fine-tuning.",
                UserWarning,
                stacklevel=2,
            )
            # NEW: Use opnorm clamp instead of spectral_norm (avoids numerical instability)
            # Only apply to z->z path layers (L_level), not embedding or output heads
            apply_opnorm_clamp_to_trm(
                self.inner,
                per_layer_max=1.0,
                num_power_iters=10,
                restrict_to_reasoning_layers=True,
            )
            # Apply output scaling to achieve target_Lz (only to L_level layers)
            enforce_global_contraction(
                self.inner,
                self.config.rl_target_Lz,
                restrict_to_reasoning_layers=True,
            )
            # Value head normalization: can be disabled independently of z→z contraction
            # A finite 2x2 diagnostic observed collapse with value-head
            # normalization and not with z→z contraction alone.
            if self.value_head is not None and not self.config.rl_disable_value_head_norm:
                apply_spectral_norm_to_value_head(self.value_head)
                enforce_global_contraction_on_value_head(self.value_head, self.config.rl_target_Lv)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def compute_counter_snapshot(self) -> Dict[str, int]:
        return self._compute_counters.snapshot()

    def restore_compute_counters(self, counters: object) -> None:
        self._compute_counters.restore(
            validate_model_counters(counters, name="model_compute_counters")
        )

    def reset_compute_counters(self) -> None:
        self._compute_counters.restore(ModelComputeCounters().snapshot())

    def record_action_value_evaluations(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Action-value evaluation count must be nonnegative.")
        self._compute_counters.action_values_evaluated += count

    def _record_recurrent_latent_update(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    ) -> None:
        self._compute_counters.recurrent_latent_update_calls += 1
        self._compute_counters.recurrent_latent_state_updates += int(
            carry.z_H.shape[0]
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def _coerce_plan_tensor(self, y: Any, batch: Dict[str, Any]) -> torch.Tensor:
        inputs = batch["inputs"]
        device = inputs.device
        if isinstance(y, dict):
            plan = y.get("inputs")
        else:
            plan = y
        if plan is None:
            raise ValueError("Plan tensor `y` must be provided.")
        if not torch.is_tensor(plan):
            plan_tensor = torch.as_tensor(plan, device=device)
        else:
            plan_tensor = plan.to(device)
        if plan_tensor.ndim == inputs.ndim - 1:
            plan_tensor = plan_tensor.unsqueeze(0)
        if plan_tensor.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"Plan batch dimension {plan_tensor.shape[0]} != input batch {inputs.shape[0]}"
            )
        if plan_tensor.shape[1:] != inputs.shape[1:]:
            plan_tensor = plan_tensor.view_as(inputs)
        return plan_tensor.to(dtype=inputs.dtype)

    def encode_plan(self, y: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Encode the current plan y using the same embedding path as inputs.
        """
        plan_tensor = batch.get("plan")
        if plan_tensor is None:
            plan_tensor = self._coerce_plan_tensor(y, batch)
        return self.inner._input_embeddings(plan_tensor, batch["puzzle_identifiers"])

    def _build_latent_context_with_plan(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        context = self.inner.build_latent_context(batch)
        plan_embeddings = self.encode_plan(batch["plan"], batch)
        if plan_embeddings.shape != context["input_embeddings"].shape:
            raise ValueError(
                "Plan embeddings must match input embeddings shape for latent updates."
            )
        context["plan_embeddings"] = plan_embeddings
        context["input_embeddings_with_plan"] = context["input_embeddings"] + plan_embeddings
        return context

    def _pool_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 2:
            return embeddings
        if embeddings.dim() == 3:
            return embeddings.mean(dim=1)
        return embeddings.view(embeddings.shape[0], -1)

    def _standardize_latent_batch(self, x: Any, y: Any) -> Dict[str, Any]:
        """
        Normalize the latent helper inputs to the batch dict format expected by the inner model.
        """
        if not isinstance(x, dict):
            raise TypeError("Expected `x` to be a batch dict containing `inputs` and `puzzle_identifiers`.")
        if "inputs" not in x or "puzzle_identifiers" not in x:
            missing = {"inputs", "puzzle_identifiers"} - set(x.keys())
            raise KeyError(f"Missing required keys for latent helpers: {missing}.")
        batch = dict(x)
        batch["plan"] = self._coerce_plan_tensor(y, batch)
        return batch

    def _resolve_latent_context(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns cached (or freshly computed) latent context for update_latent-style calls.
        """
        context = batch.get("_latent_context")
        if context is None:
            context = self._build_latent_context_with_plan(batch)
            batch["_latent_context"] = context
        elif "input_embeddings_with_plan" not in context:
            context = self._build_latent_context_with_plan(batch)
            batch["_latent_context"] = context
        return context

    def init_latent(self, x: Any, y: Any) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        Initialize z^(0) from (x, y).

        If z_init_encoder is enabled (rl_enable_z_init_encoder=True), this produces
        an (x,y)-dependent initialization as described in Section 4 of the paper:
            z^(0) = z_init(x, y)
        
        Otherwise, uses the learned global initialization (H_init, L_init) that does
        not depend explicitly on (x, y); the dependence enters through the first
        latent_step via the input + plan embeddings.
        """
        batch = self._standardize_latent_batch(x, y)
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        if self.z_init_encoder is not None:
            # Use (x, y)-dependent initialization
            context = self._build_latent_context_with_plan(batch)
            input_embeddings = context["input_embeddings"]
            plan_embeddings = context["plan_embeddings"]
            x_embed = self._pool_embedding(input_embeddings)
            y_embed = self._pool_embedding(plan_embeddings)
            
            # Get encoded initial latent
            z_init_encoded = self.z_init_encoder(x_embed, y_embed)  # [B, seq_len, hidden_size]
            z_init_encoded = z_init_encoded.to(device=device, dtype=self.inner.forward_dtype)
            
            # Add global initialization as a residual for stability
            # Use batch device to ensure consistency with the non-encoder path
            # H_init/L_init are [hidden_size], need to expand to [batch_size, seq_len, hidden_size]
            seq_len = z_init_encoded.shape[1]
            global_H = self.inner.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).to(device)
            global_L = self.inner.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).to(device)
            
            # Split encoded init into H and L components
            z_init_H, z_init_L = torch.split(z_init_encoded, self.config.hidden_size, dim=-1)

            z_H = global_H + z_init_H
            z_L = global_L + z_init_L

            # Project initial latent into the declared invariant ball.
            # The contraction specialization requires z^(0) in Z_inv.
            if self.config.rl_latent_projection_mode == "enabled":
                R = self.config.rl_latent_ball_radius
                assert R is not None
                z_H, z_L = self.inner._project_carry_to_ball(z_H, z_L, R)

            result = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=z_H,
                z_L=z_L,
            )
            self._compute_counters.latent_initialization_calls += 1
            self._compute_counters.latent_states_initialized += batch_size
            return result
        else:
            # Use global initialization
            empty_carry = self.inner.empty_carry(batch_size, device=device)
            reset_flag = torch.ones(batch_size, dtype=torch.bool, device=device)
            z = self.inner.reset_carry(reset_flag, empty_carry)

            # Project a supplied latent into the declared invariant ball.
            # The contraction specialization requires z^(0) in Z_inv.
            if self.config.rl_latent_projection_mode == "enabled":
                R = self.config.rl_latent_ball_radius
                assert R is not None
                z_H, z_L = self.inner._project_carry_to_ball(z.z_H, z.z_L, R)
                z = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=z_H,
                    z_L=z_L,
                )
            self._compute_counters.latent_initialization_calls += 1
            self._compute_counters.latent_states_initialized += batch_size
            return z

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
        input_embeds = context["input_embeddings_with_plan"]
        result = self.inner.latent_step(z, input_embeds, context["seq_info"])
        self._record_recurrent_latent_update(result)
        return result

    def update_latent_with_projection_info(
        self,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        y: Any,
        x: Any,
    ) -> Tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply one recurrent step and expose radial-projection diagnostics."""

        batch = self._standardize_latent_batch(x, y)
        context = self._resolve_latent_context(batch)
        input_embeds = context["input_embeddings_with_plan"]
        result = self.inner.latent_step_with_projection_info(
            z,
            input_embeds,
            context["seq_info"],
        )
        self._record_recurrent_latent_update(result[0])
        return result

    def unroll_latent(
        self,
        x: Any,
        y: Any,
        n: int,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, List[TinyRecursiveReasoningModel_ACTV1InnerCarry]]:
        """
        Run the inner recursion for n steps starting from z^(0).
        """
        # Initialize latent state from (x, y)
        z = self.init_latent(x, y)
        zs: List[TinyRecursiveReasoningModel_ACTV1InnerCarry] = [z]
        
        # Pre-compute latent context once for efficiency
        batch = self._standardize_latent_batch(x, y)
        batch["_latent_context"] = self._build_latent_context_with_plan(batch)
        
        for _ in range(n):
            # Pass pre-standardized batch directly to inner latent step to avoid
            # redundant standardization while preserving cached context
            context = self._resolve_latent_context(batch)
            input_embeds = context["input_embeddings_with_plan"]
            z = self.inner.latent_step(z, input_embeds, context["seq_info"])
            self._record_recurrent_latent_update(z)
            zs.append(z)
        return z, zs

    def eval_latent(self, x: Any, y: Any, n: int) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        Convenience wrapper used by RL code. Runs unroll_latent and returns only z^(n).
        """
        z_n, _ = self.unroll_latent(x, y, n)
        return z_n

    def continue_latent(
        self,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        x: Any,
        y: Any,
        n: int,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, List[TinyRecursiveReasoningModel_ACTV1InnerCarry]]:
        """
        Continue latent unrolling from an existing z (for persistent latent mode).
        Unlike unroll_latent which reinitializes z, this continues from the provided z.
        """
        batch = self._standardize_latent_batch(x, y)
        zs = []
        for _ in range(n):
            context = self._resolve_latent_context(batch)
            input_embeds = context["input_embeddings_with_plan"]
            z = self.inner.latent_step(z, input_embeds, context["seq_info"])
            self._record_recurrent_latent_update(z)
            zs.append(z)
        return z, zs

    def used_value(
        self,
        x: Dict[str, torch.Tensor],
        y: Any,
        n: int,
        z: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry] = None,
    ) -> Tuple[torch.Tensor, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        """
        Compute U_n(s) = V_ψ(z^(n)(s), x) for a batch.

        - x: batch dict with at least ["inputs", "puzzle_identifiers"]
        - y: plan tensor (same batch size and shape as `inputs`), used to build plan embeddings.
        - n: number of inner latent steps
        - z: optional existing latent state (for persistent mode). If None, reinitializes z.

        Returns:
            Tuple of (value, z_n) where z_n is the updated latent state after n steps.
        """
        if self.value_head is None:
            raise RuntimeError("Value head is not enabled; set rl_enable_value_head=True in the config.")

        # 1) Run latent unrolling to get z^(n)
        if z is None:
            # Episodic mode: reinitialize z from (x, y)
            z_n, _ = self.unroll_latent(x, y, n)
        else:
            # Persistent mode: continue from existing z
            z_n, _ = self.continue_latent(z, x, y, n)
        
        # Flatten z to preserve positional information
        z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)

        # 2) Build a fresh latent context to obtain input embeddings and summarize x
        batch = self._standardize_latent_batch(x, y)
        latent_context = self._build_latent_context_with_plan(batch)
        input_embeddings = latent_context["input_embeddings"]
        plan_embeddings = latent_context["plan_embeddings"]
        
        # Flatten embeddings
        x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
        y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
        
        combined_embed = torch.cat([x_embed, y_embed], dim=-1)

        # 3) Apply value head and return both value and updated z
        
        value = self.value_head(z_vec, combined_embed)
        self._compute_counters.value_api_calls += 1
        self._compute_counters.value_state_evaluations += int(z_n.z_H.shape[0])
        self._compute_counters.state_values_evaluated += int(value.numel())
        return value, z_n

    def policy_dist(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
        z: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry] = None,
    ) -> Tuple[Categorical, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        """
        Produce a Categorical distribution over edit actions given (x, y).

        - x: batch dict with at least ["inputs", "puzzle_identifiers"]
        - y: plan tensor (same batch size), currently assumed to have shape compatible with inputs.
        - n: number of inner latent steps for the evaluator.
        - action_mask: optional [B, action_dim] boolean mask.
        - z: optional existing latent state (for persistent mode). If None, reinitializes z.

        Returns:
            Tuple of (distribution, z_n) where z_n is the updated latent state after n steps.
        """

        if self.edit_policy is None:
            raise RuntimeError(
                "Policy head is not enabled; set rl_enable_policy_head=True and rl_num_actions>0 in the config."
            )

        # 1) Get z^(n) via latent unrolling
        if z is None:
            # Episodic mode: reinitialize z from (x, y)
            z_n, _ = self.unroll_latent(x, y, n)
        else:
            # Persistent mode: continue from existing z
            z_n, _ = self.continue_latent(z, x, y, n)
        
        # BUG FIX: Don't mean-pool z_H - FLATTEN to preserve positional info!
        # For Sudoku, actions are pos*vocab_size+digit, so position matters!
        # z_n.z_H shape: [B, total_seq_len, hidden_dim] where total_seq_len = seq_len + puzzle_emb_len
        z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)  # [B, total_seq_len * hidden_dim]

        # 2) Get x and y embeddings - FLATTEN instead of mean pool!
        batch = self._standardize_latent_batch(x, y)
        latent_context = self._build_latent_context_with_plan(batch)
        # NOTE: When puzzle_emb_ndim > 0, shape is [B, puzzle_emb_len + seq_len, hidden_dim]
        # otherwise shape is [B, seq_len, hidden_dim]. We call this "total_seq_len".
        input_embeddings = latent_context["input_embeddings"]  # [B, total_seq_len, hidden_dim]
        plan_embeddings = latent_context["plan_embeddings"]    # [B, total_seq_len, hidden_dim]
        
        # Flatten to preserve positional information
        x_embed = input_embeddings.view(input_embeddings.shape[0], -1)  # [B, total_seq_len * hidden_dim]
        y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)    # [B, total_seq_len * hidden_dim]

        dist = self.edit_policy(z_vec, x_embed, y_embed, action_mask=action_mask)
        self._compute_counters.policy_api_calls += 1
        self._compute_counters.policy_state_evaluations += int(z_n.z_H.shape[0])
        self._compute_counters.action_logits_evaluated += int(dist.logits.numel())
        return dist, z_n

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
