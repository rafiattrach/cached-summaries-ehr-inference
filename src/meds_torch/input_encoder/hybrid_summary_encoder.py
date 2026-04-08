import dataclasses
import enum
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn
import os, atexit, json, time, re

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


class HybridSummaryEncoder(nn.Module, Module):
    """Triplet-style encoder with optional single-token summary injection at the first dynamic position.

    Notes for smoke testing:
    - Summary vector can be loaded from a cache if available, otherwise a simple fallback is used
      to validate the wiring without any offline precompute.
    - Time/value channels at the summary position are zeroed to avoid fabricating per-event semantics.
    - Logs key details for easy inspection.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Triplet embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size, embedding_dim=cfg.token_dim)
        self.numeric_value_embedder = CVE(cfg)

        # Summary config (optional)
        self.summary_enabled: bool = bool(getattr(cfg, "summary", {}).get("enabled", False))
        self.summary_variant: str = str(getattr(cfg, "summary", {}).get("variant", "recent"))
        self.summary_S: int = int(getattr(cfg, "summary", {}).get("S", 256))
        self.summary_policy: str = str(getattr(cfg, "summary", {}).get("policy", "omit_if_L_lt_N"))
        self.cache_root: Optional[str] = getattr(cfg, "summary", {}).get("cache_root", None)
        self.summary_strict: bool = bool(getattr(cfg, "summary", {}).get("strict", False))
        self.summary_task_name: Optional[str] = getattr(cfg, "summary", {}).get("task_name", None)
        # Summary application mode: "token" (default single-token injection), "film" (FiLM modulation), "gate" (additive gate)
        self.summary_apply: str = str(getattr(cfg, "summary", {}).get("apply", "token"))

        # Small heads for FiLM / gate (no change to caches; operates on projected token_dim)
        if self.summary_apply == "film":
            self.film_gamma = nn.Linear(self.cfg.token_dim, self.cfg.token_dim, bias=True)
            self.film_beta = nn.Linear(self.cfg.token_dim, self.cfg.token_dim, bias=True)
        elif self.summary_apply == "gate":
            self.gate = nn.Linear(self.cfg.token_dim, self.cfg.token_dim, bias=True)

        # Trainable projection for cached summaries when their dim != token_dim
        self.summary_proj: Optional[nn.Linear] = None

        # Lightweight stats (in-memory)
        self._stats = {
            "batches": 0,
            "examples": 0,
            "attempted": 0,          # use_summary True
            "injected": 0,           # actually injected
            "skipped_policy": 0,     # omitted due to policy
            "source_cache": 0,       # came from cache
            "source_fallback": 0,    # used fallback
            "proj_used": 0,          # projection applied
        }
        self._stats_cfg = {
            "variant": self.summary_variant,
            "S": self.summary_S,
            "policy": self.summary_policy,
            "cache_root": self.cache_root,
            "token_dim": int(self.cfg.token_dim),
            "N": int(self.cfg.max_seq_len),
            "task_name": self.summary_task_name,
            "apply": self.summary_apply,
        }
        self._stats_fp: Optional[str] = os.getenv("HYBRID_SUMMARY_STATS_PATH", None)
        if self._stats_fp:
            def _dump():
                try:
                    payload = {
                        "cfg": self._stats_cfg,
                        "counts": self._stats,
                        "time_end_epoch": int(time.time()),
                    }
                    os.makedirs(os.path.dirname(self._stats_fp), exist_ok=True)
                    with open(self._stats_fp, "w") as f:
                        json.dump(payload, f)
                except Exception as e:
                    print(f"[HybridSummary] WARNING: failed stats dump to {self._stats_fp}: {e}")
            atexit.register(_dump)

    def embed_func(self, embedder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def _compute_triplet_embedding(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static_mask = batch["static_mask"]  # (B, T)
        code = batch["code"]  # (B, T)
        numeric_value = batch["numeric_value"]  # (B, T)
        time_delta_days = batch["time_delta_days"]  # (B, T)
        numeric_value_mask = batch["numeric_value_mask"]  # (B, T)

        # Embed components
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(dim=1)

        # Sum channels → (B, C, T)
        embedding = time_emb + code_emb + val_emb
        return embedding, time_emb, code_emb, val_emb

    def _first_dynamic_index(self, static_mask_row: torch.Tensor) -> int:
        # static_mask_row: (T,)
        dyn = (~static_mask_row).nonzero(as_tuple=False)
        if dyn.numel() == 0:
            return 0
        return int(dyn[0].item())

    def _sanitize_task(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or None

    def _maybe_inject_summary(self, batch: dict, embedding: torch.Tensor, time_emb: torch.Tensor, code_emb: torch.Tensor, val_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.summary_enabled:
            return embedding, time_emb, code_emb, val_emb

        # Extract required fields
        mask = batch["mask"]  # (B, T)
        static_mask = batch["static_mask"]  # (B, T)
        B, C, T = embedding.shape
        N = int(self.cfg.max_seq_len)

        # We attempt to read global end index (timestamp index) from dataset if available
        end_idx_ts: Optional[torch.Tensor] = batch.get("end_idx", None)
        subject_ids: Optional[torch.Tensor] = batch.get("subject_id", None)

        task_subdir = self._sanitize_task(self.summary_task_name)

        for b in range(B):
            # Derive cutoff in timestamps and id
            sid = int(subject_ids[b].item()) if subject_ids is not None else -1
            cutoff_ts = int(end_idx_ts[b].item()) if end_idx_ts is not None else -1

            # Decide policy based on dataset-provided length proxy
            L_est = int(mask[b].sum().item())
            use_summary = (L_est >= N) if self.summary_policy == "omit_if_L_lt_N" else True

            # Compute indices for logging (tail and window)
            tail_len = max(0, min(N - 1, max(0, cutoff_ts) if cutoff_ts >= 0 else L_est - 1))
            tail_start = max(0, (cutoff_ts if cutoff_ts >= 0 else L_est) - tail_len)
            if self.summary_variant == "recent":
                win_st_log = max(0, tail_start - self.summary_S)
                win_en_log = tail_start
            else:  # distant
                win_st_log = 0
                win_en_log = min(self.summary_S, tail_start)

            first_dyn = self._first_dynamic_index(static_mask[b])

            # Emit verbose logs
            mask_len = int(mask[b].sum().item())
            pad = int(T - mask_len)
            print(f"[HybridSummary] b={b} sid={sid} cutoff_ts={cutoff_ts} N={N} T={T} mask_tokens={mask_len} pad={pad} use_summary={use_summary} variant={self.summary_variant} tail_start={tail_start} window~=[{win_st_log},{win_en_log}) first_dyn={first_dyn}")

            if not use_summary:
                self._stats["skipped_policy"] += 1
                continue

            self._stats["attempted"] += 1

            # Resolve cache path(s)
            cache_fp = None
            tried_paths = []
            if self.cache_root is not None and sid >= 0:
                if task_subdir is not None and cutoff_ts >= 0:
                    p = os.path.join(self.cache_root, self.summary_variant, f"N={N}", task_subdir, f"{sid}__end{cutoff_ts}.pt")
                    tried_paths.append(p)
                    if os.path.exists(p):
                        cache_fp = p
                if cache_fp is None:
                    # legacy path (task-agnostic)
                    p = os.path.join(self.cache_root, self.summary_variant, f"N={N}", f"{sid}.pt")
                    tried_paths.append(p)
                    if os.path.exists(p) and not self.summary_strict:
                        cache_fp = p

            summary_vec = None
            summary_source = "fallback"
            proj_used = False

            if cache_fp is not None:
                try:
                    loaded = torch.load(cache_fp, map_location=embedding.device)
                    if isinstance(loaded, dict) and "summary" in loaded:
                        # strict validation of metadata when task-aware expected
                        if self.summary_strict and task_subdir is not None and cutoff_ts >= 0:
                            c_task = loaded.get("task_name", None)
                            c_end = int(loaded.get("end_idx_ts", -1))
                            if (c_task is None) or (self._sanitize_task(c_task) != task_subdir) or (c_end != cutoff_ts):
                                raise RuntimeError(f"[HybridSummary] STRICT: cache metadata mismatch for sid={sid} cutoff_ts={cutoff_ts} in {cache_fp}")
                        summary_vec = loaded["summary"].to(embedding.device)
                    elif isinstance(loaded, torch.Tensor):
                        if self.summary_strict and task_subdir is not None and cutoff_ts >= 0:
                            raise RuntimeError(f"[HybridSummary] STRICT: tensor cache without metadata at {cache_fp}")
                        summary_vec = loaded.to(embedding.device)
                    summary_source = "cache"
                except Exception as e:
                    if self.summary_strict:
                        raise
                    print(f"[HybridSummary] WARNING: Failed to load cache at {cache_fp}: {e}")
                    summary_vec = None

            if summary_vec is None:
                if self.summary_strict and use_summary:
                    tried = ",".join(tried_paths)
                    raise RuntimeError(
                        f"[HybridSummary] STRICT mode: missing cache for sid={sid} variant={self.summary_variant} N={N} tried=[{tried}]"
                    )
                dyn_mask = (~static_mask[b]).unsqueeze(0).unsqueeze(0)  # (1,1,T)
                dyn_emb = embedding[b : b + 1] * dyn_mask
                denom = max(1, int(dyn_mask.sum().item()))
                summary_vec = dyn_emb.sum(dim=2) / denom  # (1, C)
                proj_used = False
                self._stats["source_fallback"] += 1
            else:
                if summary_vec.dim() == 1:
                    summary_vec = summary_vec.unsqueeze(0)
                in_dim = summary_vec.shape[-1]
                if in_dim != self.cfg.token_dim:
                    if (self.summary_proj is None) or (self.summary_proj.in_features != in_dim) or (
                        self.summary_proj.out_features != self.cfg.token_dim
                    ):
                        self.summary_proj = nn.Linear(in_dim, self.cfg.token_dim)
                        self.summary_proj.to(embedding.device)
                    summary_vec = self.summary_proj(summary_vec)
                    proj_used = True
                self._stats["source_cache"] += 1

            # Apply summary according to selected mode
            if self.summary_apply == "token":
                # Original single-token injection at first dynamic position; zero time/value channels there
                code_emb[b, :, first_dyn] = summary_vec[0]
                time_emb[b, :, first_dyn] = 0
                val_emb[b, :, first_dyn] = 0
                embedding[b, :, first_dyn] = code_emb[b, :, first_dyn] + time_emb[b, :, first_dyn] + val_emb[b, :, first_dyn]
                apply_mode = "token"
            elif self.summary_apply == "film":
                # FiLM modulation across all time steps using the projected summary
                s_proj = summary_vec[0]  # (C,)
                gamma = 1.0 + torch.tanh(self.film_gamma(s_proj))  # (C,)
                beta = self.film_beta(s_proj)  # (C,)
                # Compute a modulated code embedding without in-place modification
                new_code = code_emb[b] * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
                # Optionally modulate other channels; keep as-is for minimal change
                embedding[b] = time_emb[b] + new_code + val_emb[b]
                apply_mode = "film"
            elif self.summary_apply == "gate":
                # Additive gated injection across all time steps
                s_proj = summary_vec[0]
                g = torch.sigmoid(self.gate(s_proj))  # (C,)
                new_code = code_emb[b] + (g * s_proj).unsqueeze(-1)
                embedding[b] = time_emb[b] + new_code + val_emb[b]
                apply_mode = "gate"
            else:
                # Fallback to token behavior if unknown
                code_emb[b, :, first_dyn] = summary_vec[0]
                time_emb[b, :, first_dyn] = 0
                val_emb[b, :, first_dyn] = 0
                embedding[b, :, first_dyn] = code_emb[b, :, first_dyn] + time_emb[b, :, first_dyn] + val_emb[b, :, first_dyn]
                apply_mode = "token"

            if proj_used:
                self._stats["proj_used"] += 1
            self._stats["injected"] += 1

            sv_norm = float(summary_vec.norm(p=2).item())
            shape = tuple(summary_vec.shape)
            dtype = str(summary_vec.dtype)
            dyn_ok = bool((~static_mask[b])[first_dyn].item()) if first_dyn < T else False
            mask_first = int(mask[b, first_dyn].item()) if first_dyn < T else -1
            print(
                f"[HybridSummary] b={b} apply={apply_mode} injected_at={first_dyn} dyn_ok={dyn_ok} mask_first={mask_first} summary_norm={sv_norm:.4f} "
                f"source={summary_source} proj={proj_used} cache_fp={cache_fp} shape={shape} dtype={dtype}"
            )

        return embedding, time_emb, code_emb, val_emb

    def forward(self, batch: dict) -> dict:
        embedding, time_emb, code_emb, val_emb = self._compute_triplet_embedding(batch)
        # Update lightweight stats
        try:
            self._stats["batches"] += 1
            self._stats["examples"] += int(embedding.shape[0])
        except Exception:
            pass
        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            # Dataset should have truncated to max_seq_len already; keep the check for safety
            print(f"[HybridSummary] WARNING: embedding length {embedding.shape[-1]} > max_seq_len {self.cfg.max_seq_len}")
        # Optional summary injection
        embedding, time_emb, code_emb, val_emb = self._maybe_inject_summary(batch, embedding, time_emb, code_emb, val_emb)
        # Return in standard interface
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding.transpose(1, 2)
        return batch 