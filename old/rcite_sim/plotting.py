from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _plot_metric(ax, x, y2, ylabel, logy=False, labels=("ITE", "rITE"), colors=("tab:blue", "tab:orange")):
    for i in range(2):
        ax.plot(x, y2[:, i], label=labels[i], color=colors[i])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()


def make_all_figures(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()

    betas = payload["betas"]
    exact = payload["exact"]
    ideal = payload["circuit_ideal"]
    noisy = payload["noisy"]
    sample_complexity = payload["sample_complexity"]
    cfg = payload["config"]
    spin_idx = cfg["spin_index"]

    # Figure 1: exact + ideal-circuit comparison
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    _plot_metric(axs[0, 0], betas, exact["energy"], r"Energy $\langle H \rangle$")
    axs[0, 0].set_title("Exact (matrix)")
    _plot_metric(axs[0, 1], betas, exact["spincov"], rf"Spin covariance $\langle X_{spin_idx}X_{{{spin_idx+1}}}\rangle_c$")
    _plot_metric(axs[1, 0], betas, exact["trd"], r"Trace distance to ground", logy=True)
    _plot_metric(axs[1, 1], betas, ideal["trd"], r"Trace distance (ideal circuit)", logy=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exact_and_ideal.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_exact_and_ideal.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: noise sweep for requested observables
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [("energy", r"Energy $\langle H \rangle$", False), ("spincov", rf"Spin covariance at {spin_idx}", False), ("trd", "Trace distance", True)]
    styles = ["-", "--"]
    for ax, (k, yl, logy) in zip(axs, metrics):
        for p, color in zip(sorted(noisy.keys()), plt.cm.viridis(np.linspace(0.2, 0.9, len(noisy)))):
            vals = noisy[p][k]
            for j, name in enumerate(("ITE", "rITE")):
                ax.plot(betas, vals[:, j], styles[j], color=color, label=f"{name}, p={p:g}")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(yl)
        if logy:
            ax.set_yscale("log")
        ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_noise_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_noise_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: sample complexity vs beta (target trace distance)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ideal_cost = sample_complexity["ideal"]
    ax.plot(betas, ideal_cost[:, 0], label="ITE (ideal)", color="tab:blue")
    ax.plot(betas, ideal_cost[:, 1], label="rITE (ideal)", color="tab:orange")
    for p, color in zip(sorted(noisy.keys()), plt.cm.magma(np.linspace(0.3, 0.8, len(noisy)))):
        c = sample_complexity[f"noise_{p:g}"]
        ax.plot(betas, c[:, 0], ":", color=color, label=f"ITE p={p:g}")
        ax.plot(betas, c[:, 1], "--", color=color, label=f"rITE p={p:g}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(rf"Estimated shots for $\epsilon_{{tr}} \le {cfg['target_trace_distance']}$")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sample_complexity.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_sample_complexity.png", bbox_inches="tight")
    plt.close(fig)

