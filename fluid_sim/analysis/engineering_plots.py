from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_streamwise_pressure_profile(
    x: np.ndarray,
    mean_pressure: np.ndarray,
    cp: np.ndarray,
    *,
    upstream_x: float,
    downstream_x: float,
    filename: str | Path,
) -> None:
    """Save cross-section-averaged pressure and Cp as functions of x."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_pressure, ax_cp) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    ax_pressure.plot(x, mean_pressure, linewidth=1.8)
    ax_pressure.axvline(upstream_x, linestyle="--", linewidth=1.0, label="Upstream sample")
    ax_pressure.axvline(downstream_x, linestyle=":", linewidth=1.0, label="Downstream sample")
    ax_pressure.set_ylabel("Mean pressure (Pa)")
    ax_pressure.set_title("Streamwise Pressure Profile")
    ax_pressure.grid(True, alpha=0.3)
    ax_pressure.legend()

    ax_cp.plot(x, cp, linewidth=1.8)
    ax_cp.axvline(upstream_x, linestyle="--", linewidth=1.0)
    ax_cp.axvline(downstream_x, linestyle=":", linewidth=1.0)
    ax_cp.set_xlabel("x (m)")
    ax_cp.set_ylabel("Mean pressure coefficient, Cp")
    ax_cp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
