from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rcite_sim.core import ExperimentConfig, run_all_experiments
from rcite_sim.plotting import make_all_figures


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rITE/ITE simulations and save data + paper figures.")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--g", type=float, default=1.4)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--beta-min", type=float, default=0.5)
    p.add_argument("--beta-max", type=float, default=5.0)
    p.add_argument("--beta-num", type=int, default=10)
    p.add_argument("--M-exact", type=int, default=500)
    p.add_argument("--M-circuit", type=int, default=30)
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--trotter-steps", type=int, default=10)
    p.add_argument("--trotter-order", type=int, default=2)
    p.add_argument("--spin-index", type=int, default=2)
    p.add_argument("--target-trace-distance", type=float, default=0.05)
    p.add_argument("--noise-levels", type=float, nargs="*", default=[0.0, 1e-4, 1e-3])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="paper")
    return p.parse_args()


def main() -> None:
    a = _args()
    out_root = Path(a.out).resolve()
    data_dir = out_root / "data"
    fig_dir = out_root / "figs"

    cfg = ExperimentConfig(
        n=a.n,
        J=a.J,
        g=a.g,
        beta_min=a.beta_min,
        beta_max=a.beta_max,
        beta_num=a.beta_num,
        spin_index=a.spin_index,
        M_exact=a.M_exact,
        M_circuit=a.M_circuit,
        shots=a.shots,
        trotter_steps=a.trotter_steps,
        trotter_order=a.trotter_order,
        noise_levels=tuple(a.noise_levels),
        target_trace_distance=a.target_trace_distance,
        seed=a.seed,
    )
    payload = run_all_experiments(cfg, data_dir)
    make_all_figures(payload, fig_dir)
    print("Saved data:", data_dir / "results.npz")
    print("Saved figures:", ", ".join(str(p) for p in sorted(fig_dir.glob("fig_*.*"))))


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    main()

