# Input Encoders

This folder contains the input encoder implementations used in the experiments.

## Files

### hybrid_summary_encoder.py

Extends the standard triplet encoder with an optional summary conditioning pathway. Rather than treating all patient history as direct input tokens, `HybridSummaryEncoder` loads a pre-computed fixed-size summary vector from disk and uses it to condition how recent events are represented.

Two integration modes are implemented and compared:

- **FiLM**: the summary vector is projected to per-channel scale (gamma) and shift (beta) parameters, applied element-wise to all code embeddings before the transformer. Historical context shapes the representation of recent events rather than competing with them for attention.
- **Token injection**: the summary vector is placed at the first sequence position, with time and value channels zeroed. Simpler wiring but structurally limits how much the summary can influence event representations downstream.

Summary vectors are pre-computed by `utils/precompute_long_context_summaries.py` and cached per patient. When `strict=False` and no cache file is found, the encoder falls back to a mean-pooled embedding of the current batch, which is useful for verifying the wiring without requiring pre-computed data.

### triplet_encoder.py

The baseline encoder. Each event is embedded as the sum of independently learned code, time delta, and numeric value embeddings. No summary conditioning. Used as the comparison condition across all N and S configurations.
