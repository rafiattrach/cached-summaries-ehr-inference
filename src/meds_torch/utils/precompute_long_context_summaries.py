import argparse
import os
import time
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import pickle
from pathlib import Path

# Enable TF32 matmul on supported GPUs for faster FP32
try:
	torch.set_float32_matmul_precision("high")
except Exception:
	pass


def load_code_descriptions(codes_fp: str) -> pd.DataFrame:
	return pd.read_parquet(codes_fp)


def load_code_descriptions_from_csv(csv_fp: str) -> pd.DataFrame:
	return pd.read_csv(csv_fp)


def collect_subject_sequences(data_dir: str) -> pd.DataFrame:
	# Expect triplet_tensors/metadata/codes.parquet for code map and per-subject lengths under fit_vocabulary_indices or normalization
	meta_fp = os.path.join(data_dir, "metadata", "codes.parquet")
	codes = pd.read_parquet(meta_fp)
	# Subject list from tokenization directories (filter_subjects/train etc.). Read IDs from any split available.
	subject_ids: List[int] = []
	for split in ["train", "tuning", "held_out"]:
		split_dir = os.path.join(data_dir, "filter_subjects", split)
		if not os.path.isdir(split_dir):
			continue
		for fn in os.listdir(split_dir):
			if fn.endswith(".parquet"):
				try:
					df = pd.read_parquet(os.path.join(split_dir, fn), columns=["subject_id"])  # small read
					subject_ids.extend(df["subject_id"].astype(int).tolist())
				except Exception:
					continue
	if len(subject_ids) == 0:
		raise RuntimeError(f"No subjects found under {data_dir}/filter_subjects/*")
	return pd.DataFrame({"subject_id": sorted(set(subject_ids))})


def build_window_indices(L: int, N: int, S: int, variant: str) -> tuple:
	# Tail is last min(N-1, L) events if summary is used; window selects S events preceding tail (recent) or earliest S (distant)
	if variant not in {"recent", "distant"}:
		raise ValueError(f"Unknown variant {variant}")
	tail_len = min(N - 1, L)
	if variant == "recent":
		win_en = max(0, L - tail_len)
		win_st = max(0, win_en - S)
	else:
		win_st = 0
		win_en = min(S, max(0, L - tail_len))
	return win_st, win_en


def make_text(code_ids: List[str], code_to_desc: pd.Series) -> str:
	descs = [str(code_to_desc.get(cid, cid)) for cid in code_ids]
	return "; ".join(descs)


def _sanitize_model_tag(model_id: str) -> str:
	import re
	tag = (model_id or "").split("/")[-1].lower()
	tag = re.sub(r"[^a-z0-9]+", "_", tag).strip("_")
	return tag or "model"


def _sanitize_task_name(task_name: str) -> str:
	import re
	return re.sub(r"[^a-z0-9]+", "_", (task_name or "").lower()).strip("_") or "task"


def encode_summary(
	text: str,
	model_name: str,
	tokenizer_name: Optional[str] = None,
	tok: Optional[AutoTokenizer] = None,
	model: Optional[AutoModel] = None,
	config: Optional[AutoConfig] = None,
) -> tuple[torch.Tensor, dict]:
	metrics = {}
	t0 = time.time()
	created_tok = False
	created_model = False
	if tok is None:
		tok = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False, trust_remote_code=True)
		created_tok = True
	t1 = time.time()
	if config is None:
		config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
	if model is None:
		model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
		model.eval()
		# Move to GPU if available; use FP32 for consistency with training
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		model.to(device=device, dtype=torch.float32)
		created_model = True
	else:
		# Assume model is already on the right device/dtype
		device = next(model.parameters()).device.type
	t2 = time.time()
	with torch.no_grad():
		max_pos = getattr(config, 'max_position_embeddings', None)
		if isinstance(max_pos, int) and max_pos > 0:
			max_len = max_pos
		else:
			name_lower = (model_name or "").lower()
			max_len = 512 if ('bert' in name_lower and 'modernbert' not in name_lower) else 8192
		inputs = tok(text, return_tensors='pt', truncation=True, padding=False, max_length=max_len)
		# Move inputs to device
		inputs = {k: v.to(device) for k, v in inputs.items()}
		t3 = time.time()
		outputs = model(**inputs)
		t4 = time.time()
		# Prefer pooler_output if available, else masked mean over last_hidden_state
		pooled = getattr(outputs, 'pooler_output', None)
		if pooled is None or (isinstance(pooled, torch.Tensor) and pooled.numel() == 0):
			last = outputs.last_hidden_state  # [1, T, H]
			mask = inputs['attention_mask'].unsqueeze(-1).to(last.dtype)
			sum_vec = (last * mask).sum(dim=1)
			denom = mask.sum(dim=1).clamp(min=1.0)
			pooled = sum_vec / denom
		metrics.update({
			"tok_init_ms": (t1 - t0) * 1000 if created_tok else 0.0,
			"model_init_ms": (t2 - t1) * 1000 if created_model else 0.0,
			"tokenize_ms": (t3 - t2) * 1000,
			"model_ms": (t4 - t3) * 1000,
			"pool_ms": 0.0,
			"total_ms": (t4 - t0) * 1000,
			"tokens": int(inputs['input_ids'].shape[1]),
			"device": str(device),
			"dtype": str(pooled.dtype),
		})
		return pooled.cpu(), metrics


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', required=True)
	parser.add_argument('--cache_root', required=True)
	parser.add_argument('--variant', required=True, choices=['recent', 'distant'])
	parser.add_argument('--N', type=int, required=True)
	parser.add_argument('--S', type=int, required=True)
	parser.add_argument('--model', required=True)
	parser.add_argument('--tokenizer_model', required=False)
	parser.add_argument('--subject_ids', required=False, help='Comma-separated list for a tiny smoke set')
	parser.add_argument('--policy', required=False, default='omit_if_L_lt_N', choices=['omit_if_L_lt_N','always'], help='Summary policy for precompute (mirrors encoder)')
	parser.add_argument('--mapping_csv', required=False, help='Path to meds_triplet_descriptions.csv')
	parser.add_argument('--cache_root_style', required=False, default='exact', choices=['exact','by_model'], help='exact: use cache_root as-is; by_model: append model-specific suffix to cache_root')
	# New: drive subjects from file or task parquets
	parser.add_argument('--subjects_file', required=False, help='Path to newline-separated subject_ids to process (overrides others)')
	parser.add_argument('--task_parquets', nargs='*', required=False, help='One or more task parquet paths; union of subject_ids defines processing set')
	# Task-aware cutoff inputs
	parser.add_argument('--meds_cohort_dir', required=False, help='Path to MEDS_cohort directory (to read per-subject time list)')
	parser.add_argument('--task_name', required=False, help='Task name like mortality/in_icu/first_24h. Enables task-cutoff-aware windows.')
	# Cutoff source to match training behavior
	parser.add_argument('--cutoff_source', required=False, default='schemas', choices=['schemas','meds_cohort'], help='Where to read per-subject time axis: schemas (triplet_tensors/tokenization/schemas) or MEDS_cohort/data')
	args = parser.parse_args()

	# Resolve final cache root (optionally model-specific)
	if args.cache_root_style == 'by_model':
		model_tag = _sanitize_model_tag(args.model)
		cache_root = os.path.join(args.cache_root, f"cache_{model_tag}")
	else:
		cache_root = args.cache_root

	os.makedirs(cache_root, exist_ok=True)
	# Prefer mapping CSV for descriptions
	mapping_fp = args.mapping_csv or os.path.join('meds-torch', 'mapping', 'meds_triplet_descriptions.csv')
	if not os.path.isfile(mapping_fp):
		raise FileNotFoundError(f"Mapping CSV not found at {mapping_fp}")
	map_df = load_code_descriptions_from_csv(mapping_fp)
	# Build code->description Series
	if not {'code', 'description'}.issubset(set(map_df.columns)):
		raise KeyError(f"Mapping CSV must contain columns 'code' and 'description'. Found: {list(map_df.columns)}")
	code_to_desc = map_df.set_index('code')['description']

	# Determine subjects_df precedence: subjects_file > task_parquets > explicit subject_ids > fallback scan
	subjects_df: pd.DataFrame
	if args.subjects_file and os.path.isfile(args.subjects_file):
		with open(args.subjects_file) as f:
			sids = [int(x.strip()) for x in f if x.strip()]
		subjects_df = pd.DataFrame({'subject_id': sorted(set(sids))})
	elif args.task_parquets:
		try:
			import polars as pl
		except Exception as e:
			raise RuntimeError(f"polars is required to read task parquets: {e}")
		sids: set[int] = set()
		for p in args.task_parquets:
			pp = os.path.expanduser(p)
			if not os.path.isfile(pp):
				continue
			try:
				df = pl.read_parquet(pp, columns=['subject_id'])
				sids.update(df['subject_id'].to_list())
			except Exception:
				continue
		subjects_df = pd.DataFrame({'subject_id': sorted(sids)})
	elif args.subject_ids:
		subjects = [int(x) for x in args.subject_ids.split(',') if x.strip()]
		subjects_df = pd.DataFrame({'subject_id': subjects})
	else:
		subjects_df = collect_subject_sequences(args.data_dir)

	# Task-aware subdir
	task_subdir = None
	if args.task_name:
		task_subdir = _sanitize_task_name(args.task_name)

	variant_dir = os.path.join(cache_root, args.variant, f"N={args.N}")
	if task_subdir:
		variant_dir = os.path.join(variant_dir, task_subdir)
	os.makedirs(variant_dir, exist_ok=True)

	print(f"[precompute] Starting precompute variant={args.variant} N={args.N} S={args.S} model={args.model} mapping_csv={mapping_fp}", flush=True)
	print(f"[precompute] cache_root={cache_root}", flush=True)
	all_codes: List[str] = map_df['code'].astype(str).tolist()
	from statistics import mean
	import json
	start_folder = time.time()
	per_ms = []
	per_tokens = []
	skipped_due_to_policy = 0

	# Build subject -> (split,str) mapping by scanning tokenization/event_seqs across splits
	from pathlib import Path
	print(f"[precompute] Scanning event_seqs shards for subject index...", flush=True)
	subj_to_evshard: dict[int, tuple[str,str]] = {}
	for split in ("train","tuning","held_out"):
		shard_dir = Path(args.data_dir) / "tokenization" / "event_seqs" / split
		if not shard_dir.is_dir():
			continue
		_files = list(shard_dir.glob("*.parquet"))
		try:
			_pbar_idx = tqdm(total=len(_files), desc=f"Index event_seqs/{split}", dynamic_ncols=True)
		except Exception:
			_pbar_idx = None
		for p in _files:
			try:
				import polars as pl
				df = pl.read_parquet(p, columns=["subject_id"])  # fast header read
				for sid in df["subject_id"].to_list():
					subj_to_evshard[int(sid)] = (split, p.name)
			except Exception:
				continue
			finally:
				if _pbar_idx:
					_pbar_idx.update(1)
		if _pbar_idx:
			_pbar_idx.close()
	print(f"[precompute] event_seqs index built: subjects={len(subj_to_evshard)}", flush=True)

	# If task-aware: build sid -> (prediction_time, end_idx_ts) using selected cutoff source
	sid_to_endidx_ts: dict[int, int] = {}
	if args.task_name and args.meds_cohort_dir:
		import polars as pl
		# target SIDs (prefer subjects_file if provided)
		target_sids = set(subjects_df['subject_id'].astype(int).tolist())
		# Restrict splits to those where target SIDs appear in event_seqs (faster)
		allowed_splits = set()
		for sid in target_sids:
			ss = subj_to_evshard.get(int(sid))
			if ss is not None:
				allowed_splits.add(ss[0])
		if not allowed_splits:
			allowed_splits = {"train","tuning","held_out"}
		# Try to load cached sid_to_times for this task to skip slow scan
		_sid_cache_dir = Path(cache_root) / "_sid_times"
		_sid_cache_dir.mkdir(parents=True, exist_ok=True)
		task_key = f"{_sanitize_task_name(args.task_name)}__{args.cutoff_source}"
		_sid_cache_fp = _sid_cache_dir / f"{task_key}.pkl"
		sid_to_times: dict[int, list] = {}
		if _sid_cache_fp.is_file():
			try:
				with open(_sid_cache_fp, "rb") as fh:
					_cached = pickle.load(fh)
				for k, lst in _cached.items():
					ki = int(k)
					if ki in target_sids:
						lst = list(lst)
						lst.sort()
						sid_to_times[ki] = lst
				print(f"[precompute] Loaded sid_to_times cache: {len(sid_to_times)}/{len(target_sids)} for task={args.task_name} source={args.cutoff_source}", flush=True)
			except Exception as e:
				print(f"[precompute] WARN: failed loading sid_to_times cache {str(_sid_cache_fp)}: {e}", flush=True)
		
		if len(sid_to_times) < len(target_sids):
			print(f"[precompute] Building sid_to_times for {len(target_sids)} target_sids over splits={sorted(list(allowed_splits))} from {args.cutoff_source}", flush=True)
			# 1) Build subject -> time list (stop early once we have all)
			for split in sorted(list(allowed_splits)):
				if args.cutoff_source == 'schemas':
					data_dir = Path(args.data_dir) / "tokenization" / "schemas" / split
				else:
					data_dir = Path(args.meds_cohort_dir) / "data" / split
				if not data_dir.is_dir():
					continue
				_files = list(data_dir.glob("*.parquet"))
				try:
					_pbar_times = tqdm(total=len(_files), desc=f"Build sid_to_times/{split}", dynamic_ncols=True)
				except Exception:
					_pbar_times = None
				for q in _files:
					try:
						import pandas as _pandas
						df = pl.read_parquet(q, columns=["subject_id","time"])
						# Filter to target_sids only
						df = df.filter(pl.col("subject_id").is_in(list(target_sids)))
						subs = df["subject_id"].to_list()
						times = df["time"].to_list()
						for sid_val, t_val in zip(subs, times):
							sid_i = int(sid_val)
							lst = sid_to_times.get(sid_i)
							if lst is None:
								lst = []
							# t_val may be a scalar timestamp or a list of timestamps
							if isinstance(t_val, list):
								for tt in t_val:
									if tt is not None:
										lst.append(_pandas.Timestamp(tt).to_pydatetime())
							else:
								if t_val is not None:
									lst.append(_pandas.Timestamp(t_val).to_pydatetime())
							sid_to_times[sid_i] = lst
						if len(sid_to_times) >= len(target_sids):
							print(f"[precompute] Found some times at {q}; subjects_with_times={len(sid_to_times)}/{len(target_sids)}", flush=True)
					except Exception as e:
						print(f"[precompute] WARN: failed reading {q}: {e}", flush=True)
					finally:
						if _pbar_times:
							_pbar_times.update(1); _pbar_times.set_postfix(found=f"{len(sid_to_times)}/{len(target_sids)}")
				if _pbar_times:
					_pbar_times.close()
				if len(sid_to_times) >= len(target_sids):
					break
			# Sort times per subject
			for sid_i, lst in sid_to_times.items():
				lst.sort()
			# Persist cache for reuse
			try:
				with open(_sid_cache_fp, "wb") as fh:
					pickle.dump(sid_to_times, fh)
				print(f"[precompute] Saved sid_to_times cache -> {str(_sid_cache_fp)} subjects={len(sid_to_times)}", flush=True)
			except Exception as e:
				print(f"[precompute] WARN: failed saving sid_to_times cache {str(_sid_cache_fp)}: {e}", flush=True)
		print(f"[precompute] sid_to_times built: {len(sid_to_times)}/{len(target_sids)}", flush=True)
		# 2) Read task labels (all splits)
		task_root = Path(args.meds_cohort_dir) / "tasks" / args.task_name
		if not task_root.exists():
			raise FileNotFoundError(f"Task dir not found: {task_root}")
		label_rows = []
		_label_files = list(task_root.rglob("*.parquet"))
		try:
			_pbar_labels = tqdm(total=len(_label_files), desc="Read task labels", dynamic_ncols=True)
		except Exception:
			_pbar_labels = None
		for pq in _label_files:
			try:
				df = pl.read_parquet(pq, columns=["subject_id","prediction_time"])  # unify
				# Filter to target_sids only for speed
				df = df.filter(pl.col("subject_id").is_in(list(target_sids)))
				label_rows.append(df)
			except Exception:
				continue
			finally:
				if _pbar_labels:
					_pbar_labels.update(1)
		if _pbar_labels:
			_pbar_labels.close()
		if label_rows:
			labels_df = pd.DataFrame(pl.concat(label_rows).unique().to_dict(as_series=False))
		else:
			labels_df = pd.DataFrame(columns=["subject_id","prediction_time"])
		print(f"[precompute] task labels loaded: rows={len(labels_df)} unique_subjects={labels_df['subject_id'].nunique() if not labels_df.empty else 0}", flush=True)

	# Initialize tokenizer/model ONCE and reuse per subject
	print(f"[precompute] Loading tokenizer and model from: {args.tokenizer_model or args.model}", flush=True)
	tok = AutoTokenizer.from_pretrained(args.tokenizer_model or args.model, use_fast=False, trust_remote_code=True)
	config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
	model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True)
	model.eval()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device=device, dtype=torch.float32)
	print(f"[precompute] Model ready on device={device} dtype=float32", flush=True)

	# Simple iteration with progress bar
	_iter = subjects_df['subject_id'].astype(int).tolist()
	total = len(_iter)
	processed = 0
	try:
		_pbar = tqdm(total=total, desc=f"Precompute {args.variant} N={args.N}", dynamic_ncols=True)
	except Exception:
		_pbar = None

	for sid in _iter:
		processed += 1
		# Determine cache path
		if task_subdir and (args.task_name and args.meds_cohort_dir):
			end_idx_ts = None
			if sid_to_endidx_ts is not None:
				# Compute cutoff from sid_to_times if available later
				pass
			# Name will be corrected below once cutoff is known
			cache_fp = None
		else:
			cache_fp = os.path.join(variant_dir, f"{sid}.pt")
		# Default to mapping CSV list if we can't find per-subject events
		code_ids = all_codes[: min(1024, len(all_codes))]
		# Try to read the subject's event sequence codes from tokenization/event_seqs
		split_shard = subj_to_evshard.get(sid)
		codes_nested = None
		if split_shard is not None:
			try:
				import polars as pl
				split, shard_name = split_shard
				shard_fp = Path(args.data_dir) / "tokenization" / "event_seqs" / split / shard_name
				df = pl.read_parquet(shard_fp, columns=["subject_id","code"])  # code is list[list[u16]] per timestamp
				row = df.filter(pl.col("subject_id") == sid)
				if row.height == 1:
					codes_nested = row.select("code").to_series().item(0)  # list of lists of ints
					# Flatten per-timestamp code lists into event-level code ids in order
					flat_codes: List[str] = []
					for code_list in codes_nested:
						for cid in code_list:
							flat_codes.append(str(cid))
					code_ids = flat_codes
			except Exception as e:
				print(f"[precompute] ERROR: failed per-subject read for sid={sid}: {e}. Using mapping CSV proxy.", flush=True)
		L_events = len(code_ids)

		# Decide whether to use summary and compute task-aware window
		if args.task_name and args.meds_cohort_dir and (codes_nested is not None):
			# Compute one file per prediction_time row (avoid duplicates if same cutoff repeats)
			from bisect import bisect_right
			lst_times = None
			if 'sid_to_times' in locals():
				lst_times = sid_to_times.get(int(sid))
			wrote_cutoffs = set()
			if lst_times:
				# All prediction_time rows for this subject
				_sid_rows = labels_df[labels_df['subject_id'] == int(sid)] if 'labels_df' in locals() else None
				sid_pts = []
				if _sid_rows is not None and len(_sid_rows) > 0:
					try:
						sid_pts = pd.to_datetime(_sid_rows['prediction_time'].tolist()).to_pydatetime().tolist()
					except Exception:
						# Fallback to first row if conversion fails
						v = _sid_rows['prediction_time'].iloc[0]
						sid_pts = [pd.to_datetime(v).to_pydatetime()]
				else:
					# No label rows; fallback to None
					sid_pts = [None]
				# Precompute cumulative event counts per timestamp for this subject
				events_per_ts = [len(x) for x in codes_nested]
				cum = [0]
				for c in events_per_ts:
					cum.append(cum[-1] + c)
				# Iterate all prediction times for this subject
				for pt in sid_pts:
					end_idx_ts = bisect_right(lst_times, pt) if (pt is not None) else None
					cut = int(end_idx_ts) if end_idx_ts is not None else -1
					if cut in wrote_cutoffs:
						continue
					wrote_cutoffs.add(cut)
					cache_fp = os.path.join(variant_dir, f"{sid}__end{cut}.pt") if cut >= 0 else os.path.join(variant_dir, f"{sid}.pt")
					if os.path.exists(cache_fp):
						continue
					# Map cutoff in timestamps to event index window
					events_upto_cutoff = cum[min(len(cum)-1, max(0, cut))]
					tail_len_events = min(max(0, args.N - 1), max(0, events_upto_cutoff))
					cutoff_event_end = max(0, events_upto_cutoff - tail_len_events)
					use_summary = (cutoff_event_end > 0)
					if args.variant == 'recent':
						win_en = cutoff_event_end; win_st = max(0, win_en - args.S)
					else:
						win_st = 0; win_en = min(args.S, cutoff_event_end)
					win_st_events, win_en_events = win_st, win_en
					if (processed <= 5) or (processed % 100 == 0):
						print(f"[precompute] sid={sid} cut={cut} window=[{win_st_events},{win_en_events})", flush=True)
					if not use_summary:
						skipped_due_to_policy += 1
						continue
					text = make_text(code_ids[win_st_events:win_en_events], code_to_desc)
					emb, metrics = encode_summary(text, args.model, args.tokenizer_model, tok=tok, model=model, config=config)
					per_ms.append(metrics.get("total_ms", 0.0)); per_tokens.append(metrics.get("tokens", 0))
					torch.save({
						"summary": emb.squeeze(0),
						"sid": sid,
						"variant": args.variant,
						"N": args.N,
						"S": args.S,
						"model": args.model,
						"metrics": metrics,
						"mapping_csv": mapping_fp,
						"num_codes": int(max(0, win_en_events - win_st_events)),
						"units": "events",
						"task_name": args.task_name,
						"end_idx_ts": int(cut),
						"win_st_events": int(win_st_events),
						"win_en_events": int(win_en_events),
					}, cache_fp)
					if (processed <= 5) or (processed % 100 == 0):
						print(f"[precompute] WRITE sid={sid} -> {cache_fp} tokens={metrics['tokens']} total_ms={metrics['total_ms']:.1f}", flush=True)
				# done with all rows for this subject; advance progress bar
				if _pbar:
					_pbar.update(1); _pbar.set_postfix(written=len(per_ms), skipped=skipped_due_to_policy)
				continue
			# If no times found, fall through to fallback behavior below
			use_summary = (L_events >= args.N) if args.policy == 'omit_if_L_lt_N' else True
			win_st_events, win_en_events = build_window_indices(L=L_events, N=args.N, S=args.S, variant=args.variant)
			if cache_fp is None:
				cache_fp = os.path.join(variant_dir, f"{sid}.pt")
		else:
			# Fallback to non-task-aware behavior
			use_summary = (L_events >= args.N) if args.policy == 'omit_if_L_lt_N' else True
			win_st_events, win_en_events = build_window_indices(L=L_events, N=args.N, S=args.S, variant=args.variant)
			if cache_fp is None:
				cache_fp = os.path.join(variant_dir, f"{sid}.pt")

		# Only log first few and periodic progress
		should_log = (processed <= 5) or (processed % 100 == 0)
		if should_log:
			print(f"[precompute] sid={sid} L_events={L_events} N={args.N} policy={args.policy} use_summary={use_summary} task={args.task_name or 'none'} window=[{win_st_events},{win_en_events})", flush=True)

		if os.path.exists(cache_fp):
			if _pbar:
				_pbar.update(1); _pbar.set_postfix(written=len(per_ms), skipped=skipped_due_to_policy)
			continue

		if not use_summary:
			skipped_due_to_policy += 1
			if should_log:
				print(f"[precompute] SKIP sid={sid} due to policy/cutoff (insufficient pre-cutoff events)", flush=True)
			if processed % 200 == 0:
				print(f"[precompute] PROGRESS processed={processed} written={len(per_ms)} skipped={skipped_due_to_policy}", flush=True)
			if _pbar:
				_pbar.update(1); _pbar.set_postfix(written=len(per_ms), skipped=skipped_due_to_policy)
			continue
		text = make_text(code_ids[win_st_events:win_en_events], code_to_desc)
		emb, metrics = encode_summary(text, args.model, args.tokenizer_model, tok=tok, model=model, config=config)
		per_ms.append(metrics.get("total_ms", 0.0))
		per_tokens.append(metrics.get("tokens", 0))
		torch.save({
			"summary": emb.squeeze(0),
			"sid": sid,
			"variant": args.variant,
			"N": args.N,
			"S": args.S,
			"model": args.model,
			"metrics": metrics,
			"mapping_csv": mapping_fp,
			"num_codes": int(max(0, win_en_events - win_st_events)),
			"units": "events",
			"task_name": args.task_name,
			"end_idx_ts": int(end_idx_ts if ('end_idx_ts' in locals() and end_idx_ts is not None) else -1),
			"win_st_events": int(win_st_events),
			"win_en_events": int(win_en_events),
		}, cache_fp)
		if should_log:
			print(f"[precompute] WRITE sid={sid} window=[{win_st_events},{win_en_events}) -> {cache_fp} shape={tuple(emb.shape)} tokens={metrics['tokens']} total_ms={metrics['total_ms']:.1f}", flush=True)
		if processed % 100 == 0:
			print(f"[precompute] PROGRESS processed={processed} written={len(per_ms)} skipped={skipped_due_to_policy}", flush=True)
		if _pbar:
			_pbar.update(1); _pbar.set_postfix(written=len(per_ms), skipped=skipped_due_to_policy)

	# Close progress bar
	if _pbar:
		_pbar.close()

	# Write folder-level summary
	elapsed_ms = (time.time() - start_folder) * 1000.0
	summary = {
		"variant": args.variant,
		"N": args.N,
		"S": args.S,
		"model": args.model,
		"count_written": len(per_ms),
		"skipped_due_to_policy": skipped_due_to_policy,
		"avg_subject_ms": float(mean(per_ms)) if per_ms else 0.0,
		"p50_subject_ms": float(sorted(per_ms)[len(per_ms)//2]) if per_ms else 0.0,
		"avg_tokens": float(mean(per_tokens)) if per_tokens else 0.0,
		"wall_ms": float(elapsed_ms),
		"task_name": args.task_name,
		"cutoff_source": args.cutoff_source,
	}
	with open(os.path.join(variant_dir, "_summary.json"), "w") as f:
		json.dump(summary, f, indent=2)
	print(f"[precompute] folder_summary -> {os.path.join(variant_dir, '_summary.json')} {summary}", flush=True)

if __name__ == "__main__":
	main()