"""
Smoke test for the HybridSummaryEncoder.

Verifies that the encoder can be instantiated and run a forward pass
with random inputs, without requiring any MIMIC-IV data or cached summaries.

Usage:
    python smoke_test.py
"""

import sys

import torch
from omegaconf import OmegaConf


def make_cfg(**overrides):
    base = dict(
        token_dim=32,
        vocab_size=200,
        max_seq_len=16,
        summary=dict(
            enabled=False,
        ),
    )
    base.update(overrides)
    return OmegaConf.create(base)


def run_encoder(cfg, batch_size=4, seq_len=None):
    from src.meds_torch.input_encoder.hybrid_summary_encoder import HybridSummaryEncoder

    seq_len = seq_len or cfg.max_seq_len
    encoder = HybridSummaryEncoder(cfg)
    encoder.eval()

    # Minimal batch matching the triplet collator output format
    batch = {
        "code": torch.randint(0, cfg.vocab_size, (batch_size, seq_len)),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "static_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "numeric_value": torch.randn(batch_size, seq_len),
        "numeric_value_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "time_delta_days": torch.rand(batch_size, seq_len).abs(),
        "subject_id": torch.arange(batch_size, dtype=torch.long),
    }

    with torch.no_grad():
        out = encoder(batch)

    return out


def main():
    print("=" * 60)
    print("HybridSummaryEncoder smoke test")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Test 1: baseline (no summary)
    tests_total += 1
    try:
        cfg = make_cfg(summary=dict(enabled=False))
        out = run_encoder(cfg)
        from src.meds_torch.input_encoder import INPUT_ENCODER_TOKENS_KEY
        tokens = out[INPUT_ENCODER_TOKENS_KEY]
        assert tokens.shape == (4, 16, 32), f"unexpected shape {tokens.shape}"
        print(f"[PASS] baseline (no summary): output shape {tokens.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] baseline: {e}")

    # Test 2: FiLM integration with fallback (no cache file needed)
    tests_total += 1
    try:
        cfg = make_cfg(
            summary=dict(
                enabled=True,
                variant="recent",
                S=8,
                apply="film",
                policy="omit_if_L_lt_N",
                strict=False,
                cache_root=None,
                task_name=None,
            )
        )
        out = run_encoder(cfg)
        from src.meds_torch.input_encoder import INPUT_ENCODER_TOKENS_KEY
        tokens = out[INPUT_ENCODER_TOKENS_KEY]
        assert tokens.shape == (4, 16, 32), f"unexpected shape {tokens.shape}"
        print(f"[PASS] FiLM integration (fallback summary): output shape {tokens.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] FiLM: {e}")

    # Test 3: token injection with fallback
    tests_total += 1
    try:
        cfg = make_cfg(
            summary=dict(
                enabled=True,
                variant="recent",
                S=8,
                apply="token",
                policy="omit_if_L_lt_N",
                strict=False,
                cache_root=None,
                task_name=None,
            )
        )
        out = run_encoder(cfg)
        from src.meds_torch.input_encoder import INPUT_ENCODER_TOKENS_KEY
        tokens = out[INPUT_ENCODER_TOKENS_KEY]
        assert tokens.shape == (4, 16, 32), f"unexpected shape {tokens.shape}"
        print(f"[PASS] token injection (fallback summary): output shape {tokens.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] token injection: {e}")

    print("=" * 60)
    print(f"Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)

    if tests_passed < tests_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
