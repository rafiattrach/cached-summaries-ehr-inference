# Task Configuration

All experiments in this work use: **in-ICU mortality prediction** using the first 24 hours of events after ICU admission.

This task was selected because ICU patients have among the longest event sequences in MIMIC-IV (median 7,351 events per patient), placing them squarely in the regime where memory-efficient inference matters. It also has a stable, well-defined cohort definition and is a standard benchmark for EHR prediction models.

## Extracting the task cohort from MIMIC-IV

```bash
MEDS_DIR=/path/to/mimic-iv/meds/

aces-cli --multirun hydra/launcher=joblib \
    data=sharded data.standard=meds \
    data.root="$MEDS_DIR/data" \
    "data.shard=$(expand_shards $MEDS_DIR/data)" \
    cohort_dir="$MEDS_DIR/tasks/" \
    cohort_name="mortality/in_icu/first_24h" \
    config_path="experiments/configs/tasks/mortality/in_icu/first_24h.yaml"
```

See the [ACES documentation](https://github.com/justin13601/ACES) for details on the task config format.
