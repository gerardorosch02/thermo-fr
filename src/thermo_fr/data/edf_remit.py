"""EDF's REMIT unavailability register (branch nuclear-hydro): every notice version with its publication time.

Source: EDF's open-data export of its REMIT "indisponibilités" register (opendata_edf__*.csv, ';' separated, utf-8-sig;
Licence Ouverte 2.0, attribution EDF), one row per notice version, with columns perimetre_juridique, perimetre_spatial,
status, identifiant, numero_de_version, nom, filiere, date_de_publication (UTC, ISO 8601), date_de_debut, date_de_fin, type,
cause, information_complementaire, puissance_maximale_mw, puissance_disponible_mw. The edf.fr "light" export holds the same
register with French headers, local-time stamps and decimal commas (load_edf_light), used only to cross-check counts.

Because every version keeps its publication time, the register as known at any moment tau can be rebuilt: for each notice
(identifiant), the latest version published at or before tau (`known_at`). `unavailable_by_hour` turns that into the
capacity unavailable per hour of a delivery day. The rules that decide which notices count are parameters, left open for the
pre-registration: statuses, notice types (the PMIN and aFRR chronicles are separate types), units (perimeter), how a notice
that covers part of an hour counts, and how overlapping notices of one unit combine.

Nothing here reads prices, and nothing is a model feature yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Paris"
NUCLEAR = "Nucléaire"
ACTIVE, INACTIVE, CANCELLED, DELETED = "Active", "Inactive", "Annulée", "Supprimée"
PLANNED, FORCED, PMIN_CHRONICLE, AFRR_CHRONICLE = "Planifiée", "Fortuite", "Chronique PMIN", "Chronique aFRR"
COLUMNS = {
    "identifiant": "notice", "numero_de_version": "version", "status": "status", "nom": "unit", "filiere": "fuel",
    "date_de_publication": "published", "date_de_debut": "start", "date_de_fin": "end", "type": "type", "cause": "cause",
    "puissance_maximale_mw": "max_mw", "puissance_disponible_mw": "available_mw",
    "perimetre_juridique": "perimeter_legal", "perimetre_spatial": "perimeter_spatial", "information_complementaire": "information",
}
LIGHT_COLUMNS = {
    "Status": "status", "Identifiant": "notice", "Numéro de version": "version", "Nom": "unit", "Filière": "fuel",
    "Date de début": "start", "Date de fin": "end", "Type": "type", "Cause": "cause", "Information complémentaire": "information",
    "Puissance maximale (MW)": "max_mw", "Puissance disponible (MW)": "available_mw", "Date de publication": "published",
}


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["version"] = frame["version"].astype(int)
    for c in ("max_mw", "available_mw"):
        frame[c] = pd.to_numeric(frame[c]).astype(float)
    frame["unavailable_mw"] = (frame["max_mw"] - frame["available_mw"]).clip(lower=0.0)
    return frame.sort_values(["notice", "version", "published"]).reset_index(drop=True)


def load_edf_remit(path) -> pd.DataFrame:
    """The open-data export, one row per notice version, columns renamed (COLUMNS) and times as UTC timestamps."""
    raw = pd.read_csv(Path(path), sep=";", encoding="utf-8-sig", dtype={"identifiant": str})
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"EDF export lacks columns {sorted(missing)}")
    frame = raw.rename(columns=COLUMNS)[list(COLUMNS.values())]
    for c in ("published", "start", "end"):
        frame[c] = pd.to_datetime(frame[c], utc=True)
    return _normalise(frame)


def load_edf_light(path, encoding: str = "cp1252") -> pd.DataFrame:
    """The edf.fr export (a title line, French headers, Paris local times, decimal commas), renamed like load_edf_remit."""
    raw = pd.read_csv(Path(path), sep=";", encoding=encoding, skiprows=1, decimal=",", dtype={"Identifiant": str})
    frame = raw.rename(columns=LIGHT_COLUMNS)[list(LIGHT_COLUMNS.values())]
    for c in ("published", "start", "end"):
        local = pd.to_datetime(frame[c])
        frame[c] = local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return _normalise(frame)


def known_at(register: pd.DataFrame, tau) -> pd.DataFrame:
    """The register as known at tau: for each notice, its latest version published at or before tau (whatever its status)."""
    tau = pd.Timestamp(tau)
    tau = tau.tz_localize("UTC") if tau.tzinfo is None else tau.tz_convert("UTC")
    seen = register[register["published"] <= tau]
    if seen.empty:
        return seen
    latest = seen.sort_values(["notice", "version", "published"]).groupby("notice", sort=False).tail(1)
    return latest.reset_index(drop=True)


def issue_time(day) -> pd.Timestamp:
    """10:05 Paris on D-1, the pre-market issue time, in UTC."""
    local = pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=5)
    return local.tz_localize(LOCAL_TZ).tz_convert("UTC")


def day_hours(day) -> pd.DatetimeIndex:
    """The UTC start of every hour of Paris delivery day D (23 or 25 on the clock-change days)."""
    start = pd.Timestamp(day).normalize().tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(day).normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def unavailable_by_hour(register: pd.DataFrame, day, tau=None, fuel: str | None = NUCLEAR, statuses=(ACTIVE,),
                        types=(PLANNED, FORCED), units=None, partial: str = "fraction", per_unit: str = "max") -> pd.Series:
    """Capacity unavailable (MW) in each hour of delivery day D, from the notices as known at tau (default 10:05 Paris on D-1).

    Options, to be fixed by the pre-registration:
    - statuses: which statuses of the latest known version count (Active; Inactive; Annulée; Supprimée).
    - types: notice types that count (Planifiée, Fortuite; the PMIN and aFRR chronicles are separate types).
    - units: a perimeter, the unit names to keep (None keeps every unit of `fuel`).
    - partial: "fraction" weights a notice by the share of the hour it covers; "any" counts it in full if it overlaps at all.
    - per_unit: how notices of one unit that overlap the same hour combine, "max" (the largest unavailability) or "sum".
    """
    tau = issue_time(day) if tau is None else tau
    notices = known_at(register, tau)
    if fuel is not None:
        notices = notices[notices["fuel"] == fuel]
    notices = notices[notices["status"].isin(statuses) & notices["type"].isin(types)]
    if units is not None:
        notices = notices[notices["unit"].isin(units)]
    hours = day_hours(day)
    starts = hours.to_numpy()
    ends = (hours + pd.Timedelta(hours=1)).to_numpy()
    if notices.empty:
        return pd.Series(0.0, index=hours, name="unavailable_mw")
    n_start = notices["start"].to_numpy()[:, None]
    n_end = notices["end"].to_numpy()[:, None]
    overlap = (np.minimum(n_end, ends[None, :]) - np.maximum(n_start, starts[None, :])) / np.timedelta64(1, "h")
    overlap = np.clip(overlap, 0.0, 1.0)
    if partial == "any":
        weight = (overlap > 0).astype(float)
    elif partial == "fraction":
        weight = overlap
    else:
        raise ValueError("partial must be 'fraction' or 'any'")
    contrib = weight * notices["unavailable_mw"].to_numpy()[:, None]
    per = pd.DataFrame(contrib, columns=hours).groupby(notices["unit"].to_numpy())
    if per_unit == "max":
        by_unit = per.max()
    elif per_unit == "sum":
        by_unit = per.sum()
    else:
        raise ValueError("per_unit must be 'max' or 'sum'")
    return by_unit.sum(axis=0).rename("unavailable_mw")


def structure_facts(register: pd.DataFrame, light: pd.DataFrame | None = None, fuel: str = NUCLEAR) -> dict:
    """Structural facts only (no prices): units, version delays, notices whose period covers their own publication, and the
    agreement of version counts with the light export."""
    nuc = register[register["fuel"] == fuel]
    ordered = nuc.sort_values(["notice", "version"])
    gaps = ordered.groupby("notice")["published"].diff().dropna().dt.total_seconds() / 3600.0
    first = ordered.groupby("notice").head(1)
    covers = (first["start"] <= first["published"]) & (first["end"] > first["published"])
    covers_all = (nuc["start"] <= nuc["published"]) & (nuc["end"] > nuc["published"])
    facts = {
        "rows": int(len(nuc)), "notices": int(nuc["notice"].nunique()), "units": int(nuc["unit"].nunique()),
        "notices_with_several_versions": int((nuc.groupby("notice")["version"].nunique() > 1).sum()),
        "status_counts": nuc["status"].value_counts().to_dict(), "type_counts": nuc["type"].value_counts().to_dict(),
        "first_publication": str(nuc["published"].min()), "last_publication": str(nuc["published"].max()),
        "hours_between_versions_quantiles": {str(q): round(float(gaps.quantile(q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "versions_published_in_the_same_minute_share": round(float((gaps < 1 / 60).mean()), 3),
        "share_first_versions_covering_their_publication": round(float(covers.mean()), 3),
        "share_versions_covering_their_publication": round(float(covers_all.mean()), 3),
        "share_first_versions_covering_their_publication_by_type": {k: round(float(v), 3) for k, v in covers.groupby(first["type"]).mean().items()},
    }
    if light is not None:
        lnuc = light[light["fuel"] == fuel]
        a = nuc.groupby("notice")["version"].nunique()
        b = lnuc.groupby("notice")["version"].nunique()
        both = a.index.intersection(b.index)
        facts["light_rows"] = int(len(lnuc))
        facts["light_notices"] = int(lnuc["notice"].nunique())
        facts["notices_only_in_opendata"] = int(len(a.index.difference(b.index)))
        facts["notices_only_in_light"] = int(len(b.index.difference(a.index)))
        facts["common_notices_with_equal_version_counts"] = f"{int((a[both] == b[both]).sum())} of {len(both)}"
    return facts
