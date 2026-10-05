"""Fit the models and write a short report: summary, charts, out-of-sample check, yearly breakdown."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ThermoModel
from .plots import response_plot


def out_of_sample(daily: pd.DataFrame) -> dict:
    """Train on all but the last 12 months, forecast the last 12 months of load.

    Month-of-year effects are used here instead of month-by-year effects, because
    the test months are unseen in training. Weather is taken as known (actuals),
    so this measures the model, not the weather forecast.
    """
    cutoff = daily.index.max() - pd.DateOffset(years=1)
    train, test = daily[daily.index <= cutoff], daily[daily.index > cutoff]
    model = ThermoModel("load_mw", fixed_effects="moy").fit(train)
    pred = model.predict(test)
    actual = test["load_mw"].to_numpy()
    mae = float(np.mean(np.abs(actual - pred)))
    return {
        "train_end": str(cutoff.date()),
        "test_days": int(len(test)),
        "mae_mw": round(mae, 0),
        "mape_pct": round(float(np.mean(np.abs(actual - pred) / actual) * 100), 2),
    }


def yearly_breakdown(daily: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Load and price gradients fitted one calendar year at a time, threshold held fixed.

    The last column expresses the price gradient as a share of that year's mean
    price, which makes years with very different price levels comparable.
    """
    rows = []
    for year, data in daily.groupby(daily.index.year):
        load = ThermoModel("load_mw").fit(data, threshold=threshold).fit_
        price = ThermoModel("price_eur_mwh").fit(data, threshold=threshold).fit_
        mean_price = float(data["price_eur_mwh"].mean())
        rows.append({
            "year": int(year),
            "days": int(len(data)),
            "load_gradient_mw_per_c": round(load.gradient, 0),
            "load_gradient_se": round(load.gradient_se, 0),
            "price_gradient_eur_mwh_per_c": round(price.gradient, 2),
            "price_gradient_se": round(price.gradient_se, 2),
            "price_r2": round(price.r2, 3),
            "mean_price_eur_mwh": round(mean_price, 1),
            "price_gradient_pct_of_mean": round(100.0 * price.gradient / mean_price, 2),
        })
    return pd.DataFrame(rows).set_index("year")


def price_excluding_peak_year(daily: pd.DataFrame, threshold: float) -> dict:
    """Pooled price gradient with the highest-priced year left out, as a robustness check."""
    mean_by_year = daily.groupby(daily.index.year)["price_eur_mwh"].mean()
    peak = int(mean_by_year.idxmax())
    fit = ThermoModel("price_eur_mwh").fit(daily[daily.index.year != peak], threshold=threshold).fit_
    return {"excluded_year": peak, "gradient_eur_mwh_per_c": round(fit.gradient, 2), "gradient_se": round(fit.gradient_se, 2)}


def run(daily: pd.DataFrame, out_dir, sources: dict | None = None) -> dict:
    """Fit, chart and summarise. `sources` (from sources.json) is recorded as-is for attribution."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    load = ThermoModel("load_mw").fit(daily)
    price = ThermoModel("price_eur_mwh").fit(daily, threshold=load.fit_.threshold)

    summary = {
        "period": f"{daily.index.min().date()} to {daily.index.max().date()}",
        "days": int(len(daily)),
        "heating_threshold_c": load.fit_.threshold,
        "load_gradient_mw_per_c": round(load.fit_.gradient, 0),
        "load_gradient_se": round(load.fit_.gradient_se, 0),
        "load_r2": round(load.fit_.r2, 3),
        "price_gradient_eur_mwh_per_c": round(price.fit_.gradient, 2),
        "price_gradient_se": round(price.fit_.gradient_se, 2),
        "price_r2": round(price.fit_.r2, 3),
        "price_excluding_peak_year": price_excluding_peak_year(daily, load.fit_.threshold),
        "out_of_sample": out_of_sample(daily),
    }
    if sources:
        summary["sources"] = sources

    yearly = yearly_breakdown(daily, load.fit_.threshold)
    yearly.to_csv(out / "yearly.csv")
    (out / "yearly.md").write_text(yearly_markdown(yearly, load.fit_.threshold), encoding="utf-8")

    response_plot(load, daily, out / "load_vs_temperature.png", "Daily mean load (GW)", scale=1000)
    response_plot(price, daily, out / "price_vs_temperature.png", "Daily mean day-ahead price (EUR/MWh)")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def to_markdown(s: dict) -> str:
    oos = s["out_of_sample"]
    ex = s.get("price_excluding_peak_year")
    return "\n".join([
        "# French thermosensitivity",
        "",
        f"Period: {s['period']} ({s['days']} days)",
        "",
        f"- Heating threshold: {s['heating_threshold_c']:.2f} °C",
        f"- Load: +{s['load_gradient_mw_per_c']:,.0f} MW per degree colder below the threshold "
        f"(s.e. {s['load_gradient_se']:,.0f}), R² {s['load_r2']}",
        f"- Price: +{s['price_gradient_eur_mwh_per_c']:.2f} EUR/MWh per degree colder "
        f"(s.e. {s['price_gradient_se']:.2f}), R² {s['price_r2']}",
        *([f"- Price excluding {ex['excluded_year']}: +{ex['gradient_eur_mwh_per_c']:.2f} EUR/MWh per degree "
           f"(s.e. {ex['gradient_se']:.2f})"] if ex else []),
        f"- Out of sample (last {oos['test_days']} days, trained to {oos['train_end']}): "
        f"MAE {oos['mae_mw']:,.0f} MW, MAPE {oos['mape_pct']}%",
        "",
        "Standard errors are conditional on the chosen threshold. Year-by-year estimates are in yearly.md.",
        "",
        *sources_section(s.get("sources")),
    ])


def yearly_markdown(yearly: pd.DataFrame, threshold: float) -> str:
    """The yearly breakdown as a markdown table."""
    lines = [
        f"# Yearly breakdown, threshold fixed at {threshold:.2f} °C",
        "",
        "| Year | Days | Load MW per °C (s.e.) | Price EUR/MWh per °C (s.e.) | Price R² | Mean price EUR/MWh | Price gradient, % of mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for year, r in yearly.iterrows():
        lines.append(
            f"| {year} | {r['days']:.0f} | {r['load_gradient_mw_per_c']:,.0f} ({r['load_gradient_se']:.0f}) "
            f"| {r['price_gradient_eur_mwh_per_c']:.2f} ({r['price_gradient_se']:.2f}) | {r['price_r2']:.3f} "
            f"| {r['mean_price_eur_mwh']:.1f} | {r['price_gradient_pct_of_mean']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def sources_section(sources: dict | None) -> list[str]:
    """Markdown lines naming each data source and its licence attribution."""
    if not sources:
        return []
    lines = ["## Data sources", ""]
    for series in ("load", "price", "temperature"):
        entry = sources.get(series)
        if entry:
            lines.append(f"- {series.capitalize()}: {entry['source']}. {entry.get('attribution', '')}".rstrip())
    return lines + [""]
