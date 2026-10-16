"""EDF REMIT register as known at tau, hourly unavailable nuclear capacity, and the weekly hydro loaders, on hand-built data."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.data.edf_remit import (ACTIVE, AFRR_CHRONICLE, CANCELLED, FORCED, INACTIVE, NUCLEAR, PLANNED, PMIN_CHRONICLE,
                                      day_hours, issue_time, known_at, load_edf_light, load_edf_remit, structure_facts,
                                      unavailable_by_hour)
from thermo_fr.data.hydro import load_a72, load_bfe_ogd17, parse_a72

D = pd.Timestamp("2025-01-15")
TAU = issue_time(D)  # 2025-01-14 09:05 UTC
HEADER = ("perimetre_juridique;perimetre_spatial;status;identifiant;numero_de_version;nom;filiere;date_de_publication;"
          "date_de_debut;date_de_fin;type;cause;information_complementaire;puissance_maximale_mw;puissance_disponible_mw")


def row(notice, version, status, unit, published, start, end, max_mw, available_mw, type_=PLANNED, fuel=NUCLEAR):
    return {"perimetre_juridique": "EDF SA + Luminus", "perimetre_spatial": "France", "status": status, "identifiant": notice,
            "numero_de_version": version, "nom": unit, "filiere": fuel, "date_de_publication": published,
            "date_de_debut": start, "date_de_fin": end, "type": type_, "cause": "x", "information_complementaire": "",
            "puissance_maximale_mw": max_mw, "puissance_disponible_mw": available_mw}


DAY_START, DAY_END = "2025-01-14T23:00:00+00:00", "2025-01-15T23:00:00+00:00"  # Paris day of 15 January


def register_rows():
    return [
        # A: unit U1 out for the day, revised before tau to half available, cancelled after tau
        row("A", 1, ACTIVE, "U1", "2025-01-10T08:00:00+00:00", DAY_START, DAY_END, 900, 0),
        row("A", 2, ACTIVE, "U1", "2025-01-13T08:00:00+00:00", DAY_START, DAY_END, 900, 450),
        row("A", 3, CANCELLED, "U1", "2025-01-14T12:00:00+00:00", DAY_START, DAY_END, 900, 900),
        # B: a second notice on U1 overlapping A (300 MW)
        row("B", 1, ACTIVE, "U1", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 900, 600),
        # C: unit U2 out from 10:30 to 12:00 Paris (09:30 to 11:00 UTC), 1000 MW
        row("C", 1, ACTIVE, "U2", "2025-01-11T08:00:00+00:00", "2025-01-15T09:30:00+00:00", "2025-01-15T11:00:00+00:00", 1300, 300),
        # E: cancelled before tau
        row("E", 1, ACTIVE, "U3", "2025-01-05T08:00:00+00:00", DAY_START, DAY_END, 1300, 0),
        row("E", 2, CANCELLED, "U3", "2025-01-09T08:00:00+00:00", DAY_START, DAY_END, 1300, 1300),
        # F: a PMIN chronicle and an aFRR chronicle on U4, G: an inactive notice on U5
        row("F", 1, ACTIVE, "U4", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 900, 700, type_=PMIN_CHRONICLE),
        row("F2", 1, ACTIVE, "U4", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 900, 850, type_=AFRR_CHRONICLE),
        row("G", 1, INACTIVE, "U5", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 900, 800, type_=FORCED),
        # H: another fuel, I: published exactly at tau, J: published one second after tau
        row("H", 1, ACTIVE, "HYDRO1", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 500, 0, fuel="Réservoir hydraulique"),
        row("I", 1, ACTIVE, "U6", TAU.isoformat(), DAY_START, DAY_END, 1000, 900),
        row("J", 1, ACTIVE, "U7", (TAU + pd.Timedelta(seconds=1)).isoformat(), DAY_START, DAY_END, 1000, 0),
    ]


def write_register(path, rows):
    frame = pd.DataFrame(rows)
    path.write_text(HEADER + "\n" + frame.to_csv(sep=";", header=False, index=False), encoding="utf-8-sig")
    return path


@pytest.fixture
def register(tmp_path):
    return load_edf_remit(write_register(tmp_path / "opendata_edf.csv", register_rows()))


def test_known_at_takes_the_latest_version_published_by_tau(register):
    seen = known_at(register, TAU).set_index("notice")
    assert seen.loc["A", "version"] == 2 and seen.loc["E", "status"] == CANCELLED
    assert "I" in seen.index and "J" not in seen.index  # at tau counts, one second later does not
    assert known_at(register, "2025-01-14T12:00:00Z").set_index("notice").loc["A", "status"] == CANCELLED
    assert known_at(register, "2017-01-01").empty


def test_unavailable_by_hour_rules(register):
    s = unavailable_by_hour(register, D)
    hours = day_hours(D)
    assert len(s) == 24 and (s.index == hours).all()
    # U1: max(450 from A v2, 300 from B) = 450; U6: 100; U2 adds 1000 x 0.5 in 10:00 Paris and 1000 in 11:00 Paris
    base = 450 + 100
    assert s.iloc[0] == pytest.approx(base)
    assert s.iloc[10] == pytest.approx(base + 500) and s.iloc[11] == pytest.approx(base + 1000) and s.iloc[12] == pytest.approx(base)
    assert unavailable_by_hour(register, D, partial="any").iloc[10] == pytest.approx(base + 1000)
    assert unavailable_by_hour(register, D, per_unit="sum").iloc[0] == pytest.approx(450 + 300 + 100)
    assert unavailable_by_hour(register, D, statuses=(ACTIVE, CANCELLED)).iloc[0] == pytest.approx(base + 0)  # E v2: 1300-1300
    assert unavailable_by_hour(register, D, statuses=(ACTIVE, INACTIVE), types=(PLANNED, FORCED)).iloc[0] == pytest.approx(base + 100)
    assert unavailable_by_hour(register, D, types=(PLANNED, FORCED, PMIN_CHRONICLE)).iloc[0] == pytest.approx(base + 200)
    assert unavailable_by_hour(register, D, types=(PLANNED, FORCED, PMIN_CHRONICLE, AFRR_CHRONICLE), per_unit="sum").iloc[0] == pytest.approx(450 + 300 + 100 + 200 + 50)
    assert unavailable_by_hour(register, D, units=["U1"]).iloc[0] == pytest.approx(450)
    assert unavailable_by_hour(register, D, fuel=None).iloc[0] == pytest.approx(base + 500)  # the hydro unit joins
    later = unavailable_by_hour(register, D, tau="2025-01-14T13:00:00Z")  # A cancelled, J known
    assert later.iloc[0] == pytest.approx(300 + 100 + 1000)


def test_clock_change_days_have_23_and_25_hours(register):
    assert len(unavailable_by_hour(register, "2025-03-30")) == 23 and len(unavailable_by_hour(register, "2025-10-26")) == 25


def test_versions_published_after_tau_change_nothing_at_tau(register, tmp_path):
    rng = np.random.default_rng(0)
    extra = []
    for i in range(40):
        published = (TAU + pd.Timedelta(minutes=int(rng.integers(1, 5000)))).isoformat()
        notice = rng.choice(["A", "B", "C", "F", "I", f"NEW{i}"])
        extra.append(row(str(notice), 10 + i, rng.choice([ACTIVE, CANCELLED, INACTIVE]), f"U{rng.integers(1, 9)}", published,
                         DAY_START, DAY_END, 1300, float(rng.uniform(0, 1300)), type_=rng.choice([PLANNED, FORCED])))
    perturbed = load_edf_remit(write_register(tmp_path / "perturbed.csv", register_rows() + extra))
    for kwargs in ({}, {"per_unit": "sum"}, {"statuses": (ACTIVE, INACTIVE, CANCELLED)}, {"partial": "any"}):
        pd.testing.assert_series_equal(unavailable_by_hour(register, D, **kwargs), unavailable_by_hour(perturbed, D, **kwargs))
    pd.testing.assert_frame_equal(known_at(register, TAU), known_at(perturbed, TAU))
    assert not unavailable_by_hour(perturbed, D, tau=TAU + pd.Timedelta(days=5)).equals(unavailable_by_hour(register, D, tau=TAU + pd.Timedelta(days=5)))


def test_light_export_and_structure_facts(register, tmp_path):
    lines = ['"Export de toutes les versions ( le 16/10/2026 à 16:42 )"',
             'Status;Identifiant;"Numéro de version";Nom;Filière;"Date de début";"Date de fin";Type;Cause;"Information complémentaire";'
             '"Puissance maximale (MW)";"Puissance disponible (MW)";"Date de publication"',
             'Active;A;1;"U1";"Nucléaire";"2025-01-15 00:00:00";"2025-01-16 00:00:00";Planifiée;"x";"";900,00;0,00;"2025-01-10 09:00:00"',
             'Active;A;2;"U1";"Nucléaire";"2025-01-15 00:00:00";"2025-01-16 00:00:00";Planifiée;"x";"";900,00;450,50;"2025-01-13 09:00:00"']
    path = tmp_path / "edf_light.csv"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp1252"))
    light = load_edf_light(path)
    assert light.loc[1, "available_mw"] == 450.5 and light.loc[0, "published"] == pd.Timestamp("2025-01-10T08:00Z")
    assert light.loc[0, "start"] == pd.Timestamp("2025-01-14T23:00Z")
    facts = structure_facts(register, light)
    assert facts["units"] == 7 and facts["light_notices"] == 1 and facts["notices_only_in_opendata"] == 8
    assert facts["common_notices_with_equal_version_counts"] == "0 of 1"  # A has 3 versions here, 2 in the light file


A72 = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <createdDateTime>{created}</createdDateTime>
  <TimeSeries><inBiddingZone_Domain.mRID codingScheme="A01">10YFR-RTE------C</inBiddingZone_Domain.mRID>
    <Period><timeInterval><start>2025-03-16T23:00Z</start><end>2025-04-06T22:00Z</end></timeInterval><resolution>P7D</resolution>
      <Point><position>1</position><quantity>100</quantity></Point>
      <Point><position>3</position><quantity>{third}</quantity></Point>
    </Period></TimeSeries></GL_MarketDocument>"""


def test_a72_weeks_cross_the_clock_change_and_repeat_missing_points(tmp_path):
    (tmp_path / "a.xml").write_text(A72.format(created="2026-10-16T19:00:00Z", third=120))
    (tmp_path / "b.xml").write_text(A72.format(created="2026-10-16T20:00:00Z", third=130))
    weeks = parse_a72(tmp_path / "a.xml")
    assert list(weeks["week_start"].dt.strftime("%Y-%m-%d")) == ["2025-03-17", "2025-03-24", "2025-03-31"]
    assert (weeks["week_start"].dt.dayofweek == 0).all() and (weeks["week_end"].dt.dayofweek == 6).all()
    assert list(weeks["stored_mwh"]) == [100.0, 100.0, 120.0] and list(weeks["repeated"]) == [False, True, False]
    both = load_a72([tmp_path / "a.xml", tmp_path / "b.xml"])
    assert len(both) == 3 and both["stored_mwh"].iloc[-1] == 130.0  # the later document wins


def test_bfe_loader(tmp_path):
    path = tmp_path / "bfe.csv"
    path.write_text("Datum,Wallis_speicherinhalt_gwh,TotalCH_speicherinhalt_gwh,TotalCH_max_speicherinhalt_gwh\n"
                    "2026-09-28,3000,7000,8800\n2026-09-21,2950,6900,8800\n")
    frame = load_bfe_ogd17(path)
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2026-09-21", "2026-09-28"] and "totalch_speicherinhalt_gwh" in frame
