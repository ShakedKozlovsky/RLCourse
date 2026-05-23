# PRD — Data Pipeline

## End-to-end flow

```
config(ticker, start, end)
        ▼
YFClient.download(ticker, start, end)
        │
        ├─▶ parquet cache hit ──▶ raw OHLCV DataFrame
        │
        └─▶ yfinance.download (via gatekeeper)
                │
                ├─▶ success ──▶ write parquet ──▶ raw OHLCV DataFrame
                │
                └─▶ failure ──▶ CSV fallback @ data/raw/{ticker}.csv
                                 │
                                 └─▶ raise if missing
        ▼
Splitter.chronological(df, 0.70, 0.15, 0.15) → (train_raw, val_raw, test_raw)
        ▼
FeatureEngineer.fit_transform(train_raw) → train_features (8 cols)
FeatureEngineer.transform(val_raw)       → val_features
FeatureEngineer.transform(test_raw)      → test_features
        ▼
Scaler.fit(train_features).transform(val_features).transform(test_features)
        ▼
WindowBuilder.build(features, window_size=30)
        → (N_windows, 30, 8) float32 numpy arrays per slice
        ▼
DataService caches the resulting arrays under data/processed/{ticker}_{config_hash}/
```

## Caching strategy

- **Raw cache:** `data/raw/{ticker}_{start}_{end}.parquet`. Indexed by Date. Snappy compression.
- **Processed cache:** `data/processed/{ticker}_{hash}/`, where `hash` is a stable hash of the relevant config keys (`window_size`, `features`, `train_pct`, `val_pct`, etc.). Contents: `train.npy`, `val.npy`, `test.npy`, `scaler.json`.
- Both caches are git-ignored. They are rebuildable from config alone.

## API gatekeeper

`shared/gatekeeper.py` wraps every yfinance call:

- Token-bucket rate limiter: 30 req/min, 500 req/hour (from `configs/rate_limits.json`).
- Retry on `requests.exceptions.HTTPError` (429, 503) with exponential backoff.
- Logs every call with timing.
- Never bypassed — even tests that don't actually hit yfinance go through it (with a mocked transport).

## Acceptance criteria

- `test_yfinance_client.py::test_cache_hit_no_network` — second call with the same args does not invoke `yfinance.download`.
- `test_yfinance_client.py::test_fallback_to_csv` — when the gatekeeper raises, the CSV fallback is loaded.
- `test_data_service.py::test_pipeline_shapes` — for AAPL 2020-01-01..2023-01-01, the train slice has the expected number of windows (calculated from row count after split + warmup).
- `test_data_service.py::test_no_leakage_in_scaler` — `Scaler.fit` is called exactly once, with train-only data; subsequent transforms on val/test do not refit.

## Why this is a separate layer

The data pipeline is the *only* layer that touches the network or filesystem. Isolating it makes every other layer (env, model, training) trivially testable offline with synthetic numpy arrays. This matches the SDK architecture principle from `PLAN.md` §1.
