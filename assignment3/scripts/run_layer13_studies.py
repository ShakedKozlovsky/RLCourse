"""Run all five Layer-13 audit-driven experiments on the real Kaggle data.

Episode budgets are tuned so the whole batch finishes in a few minutes.
For the README's headline numbers, increase the budgets in this driver
(not in the service) — the service stays generic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fitness_rl.services.experiment_studies import ExperimentStudies

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "layer13"
OUT.mkdir(parents=True, exist_ok=True)


def _save(name: str, payload: dict, started: float) -> None:
    elapsed = time.time() - started
    payload_with_meta = {"_runtime_seconds": round(elapsed, 2), **payload}
    (OUT / f"{name}.json").write_text(json.dumps(payload_with_meta, indent=2))
    print(f"  wrote {OUT / (name + '.json')}  ({elapsed:.1f}s)")


def main() -> None:
    cfg = ROOT / "configs" / "setup.json"

    # Multi-seed comparison: 5 seeds × 2 algos × 60 episodes ≈ longest
    print("[1/5] multi-seed comparison …")
    t = time.time()
    _save("multi_seed_comparison",
          ExperimentStudies(config_path=cfg, episodes=60).multi_seed_comparison(
              seeds=(0, 1, 2, 3, 4)), t)

    # Entropy sweep: 4 bonuses × A2C × 60 episodes
    print("[2/5] entropy sweep …")
    t = time.time()
    _save("entropy_sweep",
          ExperimentStudies(config_path=cfg, episodes=60).entropy_sweep(
              bonuses=(0.0, 0.01, 0.1, 0.5)), t)

    # REINFORCE chain: 3 variants × 60 episodes
    print("[3/5] REINFORCE variant chain …")
    t = time.time()
    _save("reinforce_chain",
          ExperimentStudies(config_path=cfg, episodes=60).reinforce_variant_chain(), t)

    # Gamma ablation: 4 gammas × A2C × 60 episodes
    print("[4/5] gamma ablation …")
    t = time.time()
    _save("gamma_ablation",
          ExperimentStudies(config_path=cfg, episodes=60).gamma_ablation(
              gammas=(0.9, 0.95, 0.99, 0.999)), t)

    # Masking ablation on LSTM env: 4 cells × (train LSTM + 60 ep)
    print("[5/5] masking ablation on LSTM env …")
    t = time.time()
    _save("masking_on_lstm_env",
          ExperimentStudies(config_path=cfg, episodes=60).masking_on_lstm_env(), t)

    print(f"\nall layer-13 results in {OUT}")


if __name__ == "__main__":
    main()
