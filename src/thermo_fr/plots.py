"""Charts for the report."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import ThermoModel  # noqa: E402


def response_plot(model: ThermoModel, daily, path, ylabel: str, scale: float = 1.0) -> None:
    """Calendar-adjusted target against temperature, with the fitted response."""
    data = daily.dropna(subset=[model.target, "temperature"])
    adjusted = model.adjusted(data) / scale
    grid = np.linspace(data["temperature"].min(), data["temperature"].max(), 200)
    curve = model.curve(grid, data) / scale

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["temperature"], adjusted, s=6, alpha=0.35, label="Daily, calendar-adjusted")
    ax.plot(grid, curve, linewidth=2.5, label=f"Fit, threshold {model.fit_.threshold:.2f} °C")
    ax.axvline(model.fit_.threshold, linestyle="--", linewidth=1)
    ax.set_xlabel("Population-weighted daily mean temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
