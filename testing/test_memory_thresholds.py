"""CPU-only regression tests; no production weights or GPU are used."""
import contextlib
import copy
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from toolkit.config_modules import TrainConfig
from toolkit.memory_management.thresholds import (
    MemoryThresholds, begin_memory_microbatch, cache_clear_for_microbatch,
    checkpoint_profile, meets_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "threshold_test_transformer",
    ROOT / "extensions_built_in/diffusion_models/minimax_h3/src/transformer.py",
)
transformer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transformer
SPEC.loader.exec_module(transformer)
torch.set_num_threads(1)


class ThresholdTests(unittest.TestCase):
    def test_defaults_and_config_validation(self):
        self.assertFalse(TrainConfig().memory_thresholds.active)
        cfg = TrainConfig(memory_thresholds={"gradient_checkpointing_min_tokens": 100})
        self.assertEqual(cfg.memory_thresholds.gradient_checkpointing_min_tokens, 100)
        for bad in [-1, 1.5, True, "100", None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MemoryThresholds.from_config({"gradient_checkpointing_min_tokens": bad})
        for bad in [[], {"typo": 1}]:
            with self.assertRaises(ValueError):
                MemoryThresholds.from_config(bad)

    def test_independent_thresholds_boundaries_and_masters(self):
        cfg = MemoryThresholds(10, 30, 20, 40)
        for tokens in (9, 10, 19, 20, 29, 30, 39, 40):
            result = checkpoint_profile(cfg, tokens, True, True, 3)
            self.assertEqual(result.gradient_checkpointing, tokens >= 10)
            self.assertEqual(result.save_on_cpu, tokens >= 30)
            self.assertEqual(result.group_size, 3 if tokens >= 20 else 1)
            self.assertEqual(meets_threshold(True, cfg.empty_cuda_cache_before_backward_min_tokens, tokens), tokens >= 40)
        self.assertEqual(checkpoint_profile(cfg, 1000, False, True, 3).group_size, 1)
        self.assertFalse(checkpoint_profile(cfg, 1000, False, True, 3).save_on_cpu)
        self.assertFalse(checkpoint_profile(cfg, 1000, True, False, 3).save_on_cpu)
        self.assertTrue(checkpoint_profile(MemoryThresholds(), 1, True, True, 3).save_on_cpu)

    def test_reset_cache_fallback_and_unsupported_config(self):
        model = type("Model", (), {"memory_thresholds": MemoryThresholds()})()
        cfg = MemoryThresholds(empty_cuda_cache_before_backward_min_tokens=20)
        begin_memory_microbatch(model, cfg, "minimax_h3")
        self.assertTrue(cache_clear_for_microbatch(True, cfg, model))
        model.memory_threshold_peak_tokens = 19
        self.assertFalse(cache_clear_for_microbatch(True, cfg, model))
        model.memory_threshold_peak_tokens = 20
        self.assertTrue(cache_clear_for_microbatch(True, cfg, model))
        self.assertFalse(cache_clear_for_microbatch(False, cfg, model))
        begin_memory_microbatch(model, cfg, "minimax_h3_ref2va")
        self.assertIsNone(model.memory_threshold_peak_tokens)
        for arch, compiled in (("flux", False), ("minimax_h3", True)):
            with self.assertRaises(ValueError):
                begin_memory_microbatch(model, cfg, arch, compiled)
        self.assertIsNone(begin_memory_microbatch(None, MemoryThresholds(), "flux"))


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        p = transformer.MiniMaxH3TransformerParams(
            hidden_size=12, num_layers=3, token_refiner_num_layers=1,
            num_attention_heads=1, attention_head_dim=12, ffn_hidden_size=24,
            latents_dim=2, audio_latents_dim=2, text_dim=6,
            timestep_input_dim=8, time_embed_hidden_size=12, time_embed_dim=8,
            rope_inv_freq_len=2,
        )
        self.base = transformer.MiniMaxH3Transformer(p).train()
        self.dynamic = copy.deepcopy(self.base)
        self.dynamic.enable_gradient_checkpointing()
        self.dynamic.activation_checkpoint_save_on_cpu = True
        self.dynamic.activation_checkpoint_group_size = 2
        self.policy = MemoryThresholds(10, 18, 18, 18)

    def inputs(self, video_rows, batch=1):
        # video_rows includes all target and reference rows. Include text/audio.
        seq = video_rows + 3
        tags = torch.tensor([1, 1, 2] + [0] * video_rows).repeat(batch, 1)
        return dict(
            hidden_states=torch.randn(batch, video_rows, 8),
            audio_hidden_states=torch.randn(batch, 1, 2),
            encoder_hidden_states=torch.randn(batch, 2, 6),
            row_timesteps=torch.full((batch, seq), 0.4),
            token_tags=tags,
            position_ids=torch.zeros(batch, seq, 3),
            text_indices=torch.arange(2),
            audio_indices=torch.tensor([2]),
            video_indices=torch.arange(3, seq),
        )

    def forward(self, model, data):
        with contextlib.redirect_stdout(io.StringIO()):
            a, b = model(**data)
        return a, b

    def compare_gradients(self):
        for (name, expected), (_, actual) in zip(self.base.named_parameters(), self.dynamic.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(expected.grad)
                self.assertIsNotNone(actual.grad)
                self.assertTrue(torch.isfinite(actual.grad).all())
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-5)

    def test_alternating_profiles_output_and_gradient_equivalence(self):
        # Below, above, below thresholds; include batch-size scaling and tail group.
        for rows, batch in [(4, 1), (20, 1), (10, 1), (10, 2), (4, 1)]:
            self.base.zero_grad()
            self.dynamic.zero_grad()
            begin_memory_microbatch(self.dynamic, self.policy, "minimax_h3_ref2va")
            data = self.inputs(rows, batch)
            expected = self.forward(self.base, data)
            with patch.object(transformer, "checkpoint", wraps=transformer.checkpoint) as spy:
                actual = self.forward(self.dynamic, data)
                tokens = batch * (rows + 3)
                self.assertEqual(spy.call_count, 0 if tokens < 10 else (3 if tokens >= 18 else 4))
            self.assertEqual(self.dynamic.memory_threshold_peak_tokens, tokens)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            sum(x.square().mean() for x in expected).backward()
            sum(x.square().mean() for x in actual).backward()
            self.compare_gradients()
            self.assertTrue(self.dynamic.gradient_checkpointing)
            self.assertTrue(self.dynamic.token_refiner.gradient_checkpointing)

    def test_multiple_forward_graphs_and_no_grad_cannot_overwrite_peak(self):
        for order in [(20, 4), (4, 20)]:
            self.base.zero_grad()
            self.dynamic.zero_grad()
            begin_memory_microbatch(self.dynamic, self.policy, "minimax_h3")
            expected_loss = 0
            actual_loss = 0
            for rows in order:
                data = self.inputs(rows)
                expected_loss += sum(x.square().mean() for x in self.forward(self.base, data))
                actual_loss += sum(x.square().mean() for x in self.forward(self.dynamic, data))
            with torch.no_grad():
                self.forward(self.dynamic, self.inputs(40))
            self.assertEqual(self.dynamic.memory_threshold_peak_tokens, 23)
            self.assertTrue(cache_clear_for_microbatch(True, self.policy, self.dynamic))
            # Another profile can be chosen before recomputation; old graphs stay valid.
            self.dynamic.memory_thresholds = MemoryThresholds(1000, 1000, 1000, 1000)
            expected_loss.backward()
            actual_loss.backward()
            self.compare_gradients()

    def test_zero_thresholds_preserve_existing_checkpoint_path(self):
        self.base.enable_gradient_checkpointing()
        self.base.activation_checkpoint_group_size = 2
        self.base.activation_checkpoint_save_on_cpu = True
        self.dynamic.memory_thresholds = MemoryThresholds()
        data = self.inputs(4)
        expected = self.forward(self.base, data)
        actual = self.forward(self.dynamic, data)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        sum(x.square().mean() for x in expected).backward()
        sum(x.square().mean() for x in actual).backward()
        self.compare_gradients()
        self.assertIsNone(self.dynamic.memory_threshold_peak_tokens)


if __name__ == "__main__":
    unittest.main()
