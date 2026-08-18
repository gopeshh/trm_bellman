#!/usr/bin/env fbpython
"""Data-only deserialization for authenticated checkpoint evidence.

Authenticating and sealing checkpoint bytes fixes *which* bytes a loader sees.
It does not make those bytes safe.  ``torch.load(weights_only=False)`` invokes
the callables named by the pickle's ``REDUCE`` opcodes, and every manifest
digest that describes a checkpoint lives inside the same evidence package as the
checkpoint, so anyone who can write the package can rewrite the bytes together
with every digest that describes them.  Runtime authorization authenticates the
producing *code*, not the produced bytes; ``0700`` ownership proves local
ownership, not provenance.  A write-sealed descriptor faithfully preserves
malicious bytes.

This module removes the capability instead of guarding it.  Checkpoints on an
evidence path are read with Torch's restricted weights-only unpickler, which
refuses any global that is not explicitly permitted and therefore never invokes
an attacker-chosen callable.

Historical Stage 0 checkpoints predate that rule: their replay buffers contain
two first-party dataclasses.  Republishing them in a data-only container would
change every checkpoint digest bound into ``MANIFEST.json``,
``RUN_MANIFEST.json``, ``model_state_inventory.json`` and the result
identities, which would break immutable publication and every historical
binding.  The two classes are therefore added to the restricted unpickler's
allowlist instead.

``HISTORICAL_CHECKPOINT_GLOBALS`` below records what those checkpoints actually
contain.  It was derived by parsing the pickle opcode stream of all 35 retained
Stage 0 checkpoints, across all three retained evidence roots, with
``pickletools.genops``, which reads opcodes without executing them.  It is a record of the input, not the allowlist: what must be
permitted was then established empirically by loading historical-shaped payloads
under ``weights_only=True`` and adding only what Torch refused.

What Torch refuses in this build, measured rather than assumed: the two
``rl.replay`` dataclasses, and every NumPy global.  This Torch does not permit
``numpy.ndarray``, ``numpy.dtype`` or the ndarray reconstruction hook by
default, so admitting them is a real widening of the restricted unpickler and
is justified below on its own terms rather than by deferring to Torch.

Why each addition is safe:

* ``Transition`` and ``ReplayLatent`` are plain dataclasses with no methods at
  all: no ``__reduce__``, no ``__setstate__``, no ``__init__`` body, no base
  classes.  Unpickling one performs ``object.__new__`` followed by a
  ``__dict__`` update, so permitting them adds no execution primitive.
* The NumPy entries are data constructors, not callables an attacker can aim.
  ``_reconstruct(subtype, shape, dtype)`` only allocates an empty array, and
  ``subtype`` must itself be a permitted global to appear in the pickle.
* The one way NumPy re-admits arbitrary objects is an object-dtype array, whose
  data buffer ``numpy.ndarray.__setstate__`` deserializes with an ordinary
  unpickler.  That escape is not closed by excluding ``ObjectDType`` from the
  allowlist, because ``numpy.dtype("O")`` reaches the same dtype from a plain
  string.  It is closed by Torch's restricted unpickler itself, and
  ``test_object_dtype_array_cannot_smuggle_a_nested_pickle`` and
  ``test_object_dtype_is_not_reachable_through_the_dtype_constructor`` prove
  that with live positive controls.  Excluding ``ObjectDType`` is retained as
  defence in depth, not as the control.
"""

from __future__ import annotations

import pickle
import threading
from contextlib import contextmanager
from typing import Any, Callable, IO, Iterator

import numpy
import torch

from rl.replay import ReplayLatent, Transition


# Concrete NumPy dtype descriptors the retained Stage 0 checkpoints require.
# Deliberately a short explicit list rather than everything ``numpy.dtypes``
# exposes: Torch's maintainers decline to allowlist NumPy at all because they
# do not control it, so every NumPy grant here is one this repository owns and
# must keep as narrow as the evidence allows.  ``ObjectDType`` is absent by
# construction.
_REQUIRED_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "UInt32DType",
    "Int64DType",
    "Float64DType",
)

# Exact result of a non-executing ``pickletools.genops`` scan over every
# retained Stage 0 checkpoint.  This records the input, not the allowlist: a
# concrete NumPy dtype descriptor is checked at ``BUILD`` time by class identity
# and so never appears as a GLOBAL opcode here.  Update only alongside a fresh
# scan.
HISTORICAL_CHECKPOINT_GLOBALS: tuple[str, ...] = (
    "_codecs.encode",
    "collections.OrderedDict",
    "numpy._core.multiarray._reconstruct",
    "numpy.dtype",
    "numpy.ndarray",
    "rl.replay.ReplayLatent",
    "rl.replay.Transition",
    "torch.BoolStorage",
    "torch.ByteStorage",
    "torch.FloatStorage",
    "torch.IntStorage",
    "torch.LongStorage",
    "torch._utils._rebuild_tensor_v2",
)


class UnsafeCheckpointPayloadError(RuntimeError):
    """Raised when checkpoint bytes are not a data-only payload."""


def _numpy_array_reconstructor() -> Callable[..., object]:
    """Resolve NumPy's ndarray reconstruction hook across NumPy 2.x and 1.x.

    This Torch build allowlists no NumPy global under any spelling, so this is
    a grant this repository owns outright, not a spelling fix on top of one
    Torch already made.  The module docstring justifies it.  The two spellings
    are handled only because the private module moved between NumPy 1.x
    (``numpy.core``) and 2.x (``numpy._core``).
    """

    for module_name in ("_core", "core"):
        module = getattr(numpy, module_name, None)
        multiarray = getattr(module, "multiarray", None)
        reconstruct = getattr(multiarray, "_reconstruct", None)
        if reconstruct is not None:
            return reconstruct
    raise UnsafeCheckpointPayloadError(
        "NumPy does not expose its ndarray reconstruction hook."
    )


# The only first-party globals added to Torch's restricted unpickler.  Both are
# plain dataclasses with no methods, so permitting them adds no execution
# primitive; the allowlist test asserts that property.
FIRST_PARTY_SAFE_GLOBALS: tuple[type, ...] = (ReplayLatent, Transition)

def _numpy_descriptor_types() -> tuple[type, ...]:
    """Resolve only the concrete dtype descriptors historical evidence needs.

    NumPy 2.x pickles an array's dtype as a concrete ``numpy.dtypes.*DType``
    instance, which Torch's restricted unpickler rejects at ``BUILD`` time by
    class identity rather than by global name, so these never appear in an
    opcode scan.  Only the descriptors the retained checkpoints actually use
    are granted -- the saved NumPy RNG state is a ``uint32`` array with an
    ``int64``/``float64`` tail -- rather than everything ``numpy.dtypes``
    happens to expose in the installed NumPy.

    Each is an inert descriptor type: constructing one yields a dtype and
    nothing else, and the array contents it describes are themselves unpickled
    under this same allowlist.
    """

    descriptors = getattr(numpy, "dtypes", None)
    if descriptors is None:
        return ()
    selected: list[type] = []
    for name in _REQUIRED_DESCRIPTOR_NAMES:
        candidate = getattr(descriptors, name, None)
        if isinstance(candidate, type) and issubclass(candidate, numpy.dtype):
            selected.append(candidate)
    return tuple(selected)


# NumPy's own array data types, needed for the saved NumPy RNG state.  This
# Torch build permits none of them by default, so this is a genuine widening of
# the restricted unpickler; see the module docstring for why each is inert.
# The concrete descriptor set is resolved from the installed NumPy rather than
# hard-coded, because the ``numpy.dtypes`` membership is version-dependent; the
# allowlist test pins the *property* that every member is a dtype descriptor.
NUMPY_SAFE_GLOBALS: tuple[object, ...] = (
    _numpy_array_reconstructor(),
    numpy.ndarray,
    numpy.dtype,
    *_numpy_descriptor_types(),
)

ADDITIONAL_SAFE_GLOBALS: tuple[object, ...] = (
    *FIRST_PARTY_SAFE_GLOBALS,
    *NUMPY_SAFE_GLOBALS,
)


# Torch's user allowlist is process-global, and ``import torch`` itself
# installs a large ambient set (74 entries in this build, including a plain
# aimable function).  ``torch.serialization.safe_globals`` UNIONS with whatever
# is already there and removes its own entries on exit without reference
# counting, so using it directly would (a) grant every ambient entry to a
# hostile checkpoint and (b) strip an outer scope's grants when an inner scope
# closed.  Neither is acceptable for an evidence loader that claims an exact
# allowlist, so this module manages the state itself: snapshot, clear, install
# exactly ``ADDITIONAL_SAFE_GLOBALS``, then clear and restore the snapshot.
# The lock is re-entrant for the owning thread and blocks other threads for the
# duration of a load, matching the fact that Torch's state is not thread-safe.
_SCOPE_LOCK = threading.RLock()
_SCOPE_DEPTH = 0
_REQUIRED_TORCH_ALLOWLIST_API = (
    "get_safe_globals",
    "clear_safe_globals",
    "add_safe_globals",
)


@contextmanager
def safe_checkpoint_globals() -> Iterator[None]:
    """Make the effective allowlist exactly ``ADDITIONAL_SAFE_GLOBALS``.

    Ambient grants installed by importing Torch are removed for the duration
    of the load and restored afterwards, so a hostile checkpoint cannot aim a
    callable this module never reviewed.  Re-entrant: nested scopes are a
    no-op, and the ambient snapshot is restored exactly once on both the
    success and the exception path.
    """

    global _SCOPE_DEPTH
    missing = [
        name
        for name in _REQUIRED_TORCH_ALLOWLIST_API
        if not hasattr(torch.serialization, name)
    ]
    if missing:
        raise UnsafeCheckpointPayloadError(
            f"Torch does not support allowlist isolation; missing {missing}."
        )
    with _SCOPE_LOCK:
        if _SCOPE_DEPTH:
            _SCOPE_DEPTH += 1
            try:
                yield
            finally:
                _SCOPE_DEPTH -= 1
            return
        ambient = list(torch.serialization.get_safe_globals())
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(list(ADDITIONAL_SAFE_GLOBALS))
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
            torch.serialization.clear_safe_globals()
            if ambient:
                torch.serialization.add_safe_globals(ambient)


def load_data_only_checkpoint(
    source: IO[bytes],
    *,
    map_location: str = "cpu",
) -> Any:
    """Deserialize one checkpoint without granting it code execution.

    ``source`` must be a readable binary stream positioned by the caller.  Any
    global outside Torch's weights-only set and ``ADDITIONAL_SAFE_GLOBALS``
    aborts the load before the offending object is constructed.
    """

    try:
        with safe_checkpoint_globals():
            return torch.load(
                source,
                map_location=map_location,
                weights_only=True,
            )
    except UnsafeCheckpointPayloadError:
        raise
    except pickle.UnpicklingError as exc:
        raise UnsafeCheckpointPayloadError(
            "Checkpoint payload is not data-only: it names a global outside "
            f"the authenticated allowlist ({exc})."
        ) from exc
    except (AttributeError, ImportError, RuntimeError, TypeError) as exc:
        raise UnsafeCheckpointPayloadError(
            f"Checkpoint payload could not be read as data-only bytes ({exc})."
        ) from exc
