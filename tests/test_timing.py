# Original code (CMP 3005 term project).
"""Step 6d: log-log fits, theory slopes, verdict rule, deterministic writers and the smoke run.
No assertion is made on real timings."""

import csv
import math
import time

import pytest

from experiments.timing import (N_COMPONENTS, SLOPE_TOL, build_fits, fit_loglog, main, theory_slope, verdict,
                                write_fits)

NS = [50, 100, 200, 400, 800]


def test_fit_recovers_quadratic():
    f = fit_loglog(NS, [3e-7 * n ** 2 for n in NS])
    assert f["slope"] == pytest.approx(2.0) and f["se"] == pytest.approx(0.0, abs=1e-9)
    assert f["r2"] == pytest.approx(1.0)


def test_fit_n_log_n_slope_slightly_above_one():
    f = fit_loglog(NS, [n * math.log(n) for n in NS])
    assert 1.05 <= f["slope"] <= 1.30 and f["r2"] > 0.99


def test_fit_with_two_points_has_no_se():
    f = fit_loglog([20, 40], [1.0, 4.0])
    assert f["slope"] == pytest.approx(2.0) and math.isnan(f["se"])


def test_theory_slope_accounts_for_log_factor():
    big = NS[1:]
    expected = fit_loglog(big, [n * n * math.log(n) for n in big])["slope"]
    assert theory_slope("cw", big) == pytest.approx(expected) and 2.1 < expected < 2.25
    assert theory_slope("split_k", big) == pytest.approx(2.0)
    assert theory_slope("tabu_delta", big) == pytest.approx(0.0)


def test_verdict_rule():
    assert verdict(1.30, 0.20, 1.0) == "consistent"           # inside 2*SE (0.40)
    assert verdict(1.10, 0.01, 1.0) == "consistent"           # 2*SE = 0.02 < 0.10 <= SLOPE_TOL
    assert verdict(1.30, 0.05, 1.0) == "differs"              # outside max(0.10, 0.15)
    assert verdict(1.0 + SLOPE_TOL - 1e-9, math.nan, 1.0) == "consistent"   # unknown SE: tolerance only
    assert verdict(1.20, math.nan, 1.0) == "differs"
    assert verdict(math.nan, 0.1, 1.0) == "n/a"


def _synthetic_rows():
    """Timing rows following the theory exactly (cw: n^2 log n, split_k: n^2, tabu: n),
    with a small rep-to-rep spread, plus a Q-sweep where Split time ~ L."""
    rows = []
    for n in NS:
        for rep in range(3):
            for comp, t in (("cw", 1e-8 * n * n * math.log(n)), ("split_k", 1e-8 * n * n),
                            ("tabu_iteration", 1e-5 * n)):
                rows.append({"component": comp, "n": n, "rep": rep, "seconds": t * (1 + 0.01 * rep), "Q": 100,
                             "K": n // 6 + 1, "L_eff": 6.5})
    for Q, L, K in ((50, 3.2, 63), (100, 6.4, 32), (200, 12.8, 17)):
        for rep in range(3):
            rows.append({"component": "split_unlimited_q", "n": 200, "rep": rep, "seconds": 1e-6 * L, "Q": Q,
                         "K": K, "L_eff": L})
    return rows


def test_build_fits_on_synthetic_data():
    fits = {f["component"]: f for f in build_fits(_synthetic_rows())}
    assert set(fits) == {"cw", "split_k", "tabu_iteration", "split_unlimited_q"}
    for comp in ("cw", "split_k", "tabu_iteration"):
        assert fits[comp]["verdict"] == "consistent", fits[comp]
        assert fits[comp]["fit_n100_slope"] == pytest.approx(fits[comp]["theory_slope"], abs=0.01)
    assert fits["split_unlimited_q"]["x"] == "L" and fits["split_unlimited_q"]["verdict"] == "consistent"


def test_fits_writer_is_deterministic(tmp_path):
    fits = build_fits(_synthetic_rows())
    a, b = write_fits(fits, tmp_path / "a"), write_fits(fits, tmp_path / "b")
    for pa, pb in zip(a, b):
        data = pa.read_bytes()
        assert data == pb.read_bytes() and b"\r" not in data
    assert f"max(2*SE, {SLOPE_TOL})" in (tmp_path / "a" / "complexity_fits.md").read_text()


def test_smoke_run_writes_everything(tmp_path):
    start = time.perf_counter()
    result = main(["--smoke", "--out", str(tmp_path)])
    assert time.perf_counter() - start < 60                       # a few seconds in practice
    names = {p.name for p in result["files"]}
    for name in ("timing.csv", "hardware.txt", "share_ga_ls.csv", "complexity_fits.csv", "complexity_fits.md",
                 "complexity_overview.png", "split_vs_L.png", "delta_vs_full.png"):
        assert name in names and (tmp_path / name).exists(), name
    with (tmp_path / "timing.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    components = {r["component"] for r in rows}
    assert set(N_COMPONENTS) | {"split_unlimited_q", "split_k_q"} == components
    assert {int(r["n"]) for r in rows if r["component"] == "cw"} == {20, 40}
    assert {r["rep"] for r in rows} == {"0", "1"} and all(float(r["seconds"]) > 0 for r in rows
                                                          if r["component"] != "tabu_iteration")
    assert b"\r" not in (tmp_path / "timing.csv").read_bytes()
    with (tmp_path / "share_ga_ls.csv").open(newline="") as f:
        share = list(csv.DictReader(f))
    assert len(share) == 2
    import cvrp.ga                                                   # profile timers were removed again
    assert cvrp.ga.decode.__name__ == "decode" and cvrp.ga.descent.__name__ == "descent"
    for s in share:
        total = float(s["descent_share"]) + float(s["split_decode_share"]) + float(s["rest_share"])
        assert total == pytest.approx(1.0)
