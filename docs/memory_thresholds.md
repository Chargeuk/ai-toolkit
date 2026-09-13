# MiniMax H3 per-batch memory thresholds

Optional thresholds select activation-memory settings using the actual packed transformer workload. Existing jobs behave identically when this configuration is absent or all thresholds are zero.

```yaml
train:
  gradient_checkpointing: true
  empty_cuda_cache_before_backward: true
  memory_thresholds:
    gradient_checkpointing_min_tokens: 0
    activation_checkpoint_save_on_cpu_min_tokens: 16000
    activation_checkpoint_group_size_min_tokens: 24000
    empty_cuda_cache_before_backward_min_tokens: 24000
model:
  activation_checkpoint_save_on_cpu: true
  activation_checkpoint_group_size: 2
  compile: false
```

These numbers illustrate the syntax; they are not validated recommendations for a 32 GB card.

- The workload is batch size multiplied by padded packed sequence length, including target and reference latent frames, text and audio. Padding consumes computation and is counted.
- The threshold is inclusive: the option applies when tokens are greater than or equal to its minimum. Zero retains unconditional behavior. A disabled parent switch always wins.
- Below the checkpoint-group threshold the group size is 1. CPU activation saving and grouping depend on effective gradient checkpointing.
- Forward decisions use local immutable profiles. A later forward cannot alter earlier checkpoint recomputation.
- CUDA cache clearing uses the largest differentiable forward since the microbatch began; no-gradient passes cannot overwrite it. Missing observations conservatively retain the enabled cache clearing.
- Layer placement, ConvRot backward storage, pinned buffers and dataset caching remain static. Only MiniMax H3 and Ref2VA support positive thresholds; compilation must be disabled.
- Thresholds are workload proxies, not estimated GB or guarantees against OOM. Calibrate conservatively against measured peaks; other memory consumers and attention implementations matter.
- The log records MEMORY_THRESHOLDS forward decisions and backward cache-clearing decisions whenever thresholds are active. With all-zero defaults it adds no policy logging.
- GUI controls are in the Memory card, beside the four relevant settings. Changes require a subsequent job start; they cannot change an already-running Python process.

Validation: `CUDA_VISIBLE_DEVICES="" python -m unittest testing.test_memory_thresholds -v` uses a small CPU MiniMax transformer. It checks validation, exact boundaries, independent controls, disabled switches, varying batch sizes, alternating profiles, multiple outstanding graphs, no-gradient passes, gradient equivalence and legacy defaults. It does not allocate production model weights or benchmark GPU savings.
