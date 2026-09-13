"""Opt-in MiniMax H3 memory policy. Token counts are workload proxies, not VRAM estimates."""
from dataclasses import dataclass, fields
from typing import Mapping, Optional


@dataclass(frozen=True)
class MemoryThresholds:
    gradient_checkpointing_min_tokens: int = 0
    activation_checkpoint_save_on_cpu_min_tokens: int = 0
    activation_checkpoint_group_size_min_tokens: int = 0
    empty_cuda_cache_before_backward_min_tokens: int = 0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"memory_thresholds.{field.name} must be a non-negative integer")

    @classmethod
    def from_config(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("train.memory_thresholds must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown memory_thresholds fields: {sorted(unknown)}")
        return cls(**value)

    @property
    def active(self):
        return any(getattr(self, f.name) > 0 for f in fields(self))


def meets_threshold(enabled: bool, minimum: int, tokens: int) -> bool:
    """Master switches stay authoritative. Zero preserves unconditional behavior."""
    return bool(enabled and tokens >= minimum)


@dataclass(frozen=True)
class CheckpointProfile:
    gradient_checkpointing: bool
    save_on_cpu: bool
    group_size: int


def checkpoint_profile(thresholds, tokens, checkpointing, save_on_cpu, group_size):
    checkpointing = meets_threshold(
        checkpointing, thresholds.gradient_checkpointing_min_tokens, tokens
    )
    return CheckpointProfile(
        gradient_checkpointing=checkpointing,
        save_on_cpu=checkpointing and meets_threshold(
            save_on_cpu, thresholds.activation_checkpoint_save_on_cpu_min_tokens, tokens
        ),
        group_size=group_size if checkpointing and meets_threshold(
            True, thresholds.activation_checkpoint_group_size_min_tokens, tokens
        ) else 1,
    )


def begin_memory_microbatch(model, thresholds, arch, compiled=False):
    """Reset only observations, before any forward belonging to this backward."""
    if not thresholds.active:
        return None
    if arch not in ("minimax_h3", "minimax_h3_ref2va"):
        raise ValueError("Memory thresholds currently support MiniMax H3 / Ref2VA only")
    if compiled:
        raise ValueError("Memory thresholds require compile=false")
    if not hasattr(model, "memory_thresholds"):
        raise ValueError("The training transformer does not support memory thresholds")
    model.memory_thresholds = thresholds
    model.memory_threshold_peak_tokens = None
    return model


def cache_clear_for_microbatch(enabled, thresholds, model):
    if not enabled or not thresholds.active:
        return bool(enabled)
    # No observed differentiable forward: retain the enabled safety measure.
    tokens: Optional[int] = model.memory_threshold_peak_tokens
    if tokens is None:
        return True
    return meets_threshold(
        enabled, thresholds.empty_cuda_cache_before_backward_min_tokens, tokens
    )
