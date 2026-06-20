# Reproducibility

> Bit-for-bit identical runs at the same seed. Verified by
> `tests/integration/test_reproducibility.py::test_same_seed_identical_diagnostics`.

## Replay the headline policy

```bash
cd assignment5
uv sync --extra dev
uv run roomba-lab evaluate saved_models/headline_policy.pt --n-episodes 5
```

## Re-train from scratch (~45 s CPU)

```bash
uv run python scripts/train_and_visualise.py
```

This will:

1. Train a fresh DDPG policy for 4 000 steps (seed=0)
2. Save the checkpoint to `saved_models/headline_policy.pt`
3. Emit the 4 mandatory plots to `assets/plots/`

## Re-run the noise-σ sweep (~9 min CPU)

```bash
uv run python scripts/run_noise_sigma_sweep.py
uv run python scripts/plot_sweep.py noise_sigma
```

## Re-emit the wiki

```bash
uv run roomba-lab graphify
ls docs/wiki/      # 98 nodes + 189 edges
```

## Run the full test suite (~50 s CPU)

```bash
uv run pytest -v
```

107 tests, all green.

## Update the HouseExpo sample

10 maps from the official `map_id_10.txt` shortlist are committed in
`data/raw/sample_maps/`. To pull more:

```bash
curl -L "https://github.com/TeaganLi/HouseExpo/raw/master/HouseExpo/json.tar.gz" \
     -o /tmp/houseexpo.tar.gz
tar -xzf /tmp/houseexpo.tar.gz -C /tmp/houseexpo
# Pick any <id>.json from /tmp/houseexpo/json/ and copy into data/raw/sample_maps/
```

## Bit-for-bit guarantee

`shared/seed.py::set_global_seed(seed)` seeds Python, NumPy, and PyTorch
deterministically; `torch.backends.cudnn.deterministic = True`.
`np.random.default_rng(seed)` is used throughout for buffer + env sampling.
With the same seed and the same HouseExpo sample, the training diagnostics
should match exactly.
