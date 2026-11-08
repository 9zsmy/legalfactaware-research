# LegalFactAware IN-Abs extension

This isolated experiment implements the English IN-Abs extension without
modifying any frozen CAIL artifact.

## Phase 1

```bash
python run_inabs_phase1.py \
  --data ../../data/in_abs_official_2022/dataset/IN-Abs \
  --archive ../../data/in_abs_official_2022/dataset.zip \
  --output outputs/phase1_dev
```

The program intentionally has no test-evaluation option. It writes the fixed
split, train-only budget calibration, per-case dev predictions and aggregate
metrics. Run `python -m unittest -v test_inabs_phase1.py` for unit tests.

