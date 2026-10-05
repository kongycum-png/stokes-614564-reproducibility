"""Focused Q/T and four-boundary analysis from retained refined curves.

This script performs postprocessing only.  It does not propagate fields and it
does not read fitted propagation coefficients.  Q, T, and pbar are taken from
the same input-normalized channel integrals and the same ROI for each row.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
from scipy.interpolate import CubicSpline

from observables import boundary_event, event


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "principal_refined"
DEST = ROOT / "data" / "focused_refinement"
DEST.mkdir(exist_ok=True)

BOUNDARIES = ("scaled_0.9", "scaled_1", "scaled_1.1", "actual_root")
ANALYSIS_BOUNDARIES = ("scaled_0.8", "scaled_0.9", "scaled_1", "scaled_1.1", "scaled_1.2", "actual_root")
GAMMAS = (0.3, 0.5, 0.7)
ORIGINAL_RADII = (24, 32, 40, 56, 72, 96)


def write_csv(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def threshold(event_record: dict, gamma: float) -> tuple[float | None, str]:
    if event_record.get("status") != "defined":
        return None, event_record.get("status", "unknown_event_status")
    record = event_record.get("thresholds", {}).get(f"{gamma:.1f}", {})
    value = record.get("tau")
    return value, record.get("status", "missing_threshold_record")


def value_at(tau: np.ndarray, values: np.ndarray, position: float | None) -> float | None:
    if position is None or not np.isfinite(values).all():
        return None
    return float(CubicSpline(tau, values)(position))


def difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def signal_event(tau: np.ndarray, values: np.ndarray, boundary: str, root_record: dict | None) -> dict:
    if boundary == "actual_root":
        return boundary_event(tau, values, root_record)
    return event(tau, values)


def paired_status(*items: tuple[float | None, str]) -> str:
    failures = [status for value, status in items if value is None]
    return "defined" if not failures else ";".join(failures)


def load_cases() -> list[tuple[dict, np.lib.npyio.NpzFile]]:
    cases = []
    for meta_path in sorted(SOURCE.glob("R[0-9]*_eta*.json")):
        if meta_path.name.endswith("_status.json"):
            continue
        curve_path = meta_path.with_name(meta_path.stem + "_curves.npz")
        if not curve_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("job_status") != "RUN_COMPLETED":
            continue
        cases.append((meta, np.load(curve_path)))
    return cases


def analyze() -> dict:
    event_rows: list[dict] = []
    pair_rows: list[dict] = []
    ratio_rows: list[dict] = []
    ratio_summary: list[dict] = []
    boundary_wide: list[dict] = []
    identity_max = 0.0
    cases = load_cases()

    for meta, curves in cases:
        R0 = int(meta["R0"])
        eta = float(meta["eta"])
        tau = np.asarray(curves["tau"], dtype=float)
        orders = sorted(int(key) for key in meta["events"])
        computed: dict[tuple[int, str, str], dict] = {}

        for m in orders:
            root_record = meta.get("root_tracking", {}).get(str(m))
            for boundary in ANALYSIS_BOUNDARIES:
                signals = {
                    name: np.asarray(curves[f"m{m}__{boundary}__{name}"], dtype=float)
                    for name in ("Q", "T", "pbar", "Pplus", "Pminus")
                }
                identity = np.nanmax(np.abs(signals["Q"] - signals["T"] * signals["pbar"]))
                identity_max = max(identity_max, float(identity))
                for name in ("Q", "T"):
                    record = signal_event(tau, signals[name], boundary, root_record)
                    computed[(m, boundary, name)] = record
                    row = {
                        "R0": R0,
                        "eta": eta,
                        "m": m,
                        "boundary": boundary,
                        "signal": name,
                        "event_status": record.get("status"),
                        "tau_peak": record.get("tau_peak"),
                        "peak_value_input_normalized": record.get("q_peak"),
                        "peak_curvature": record.get("peak_curvature"),
                        "missing_plane_count": record.get("missing_plane_count", 0),
                        "root_jumps": record.get("root_jumps", 0),
                        "identity_max_abs_Q_minus_Tpbar": float(identity),
                    }
                    for gamma in GAMMAS:
                        location, status = threshold(record, gamma)
                        row[f"h_{gamma:g}"] = location
                        row[f"h_{gamma:g}_status"] = status
                    if name == "Q":
                        peak = record.get("tau_peak")
                        half, _ = threshold(record, 0.5)
                        row["pbar_at_Q_peak"] = value_at(tau, signals["pbar"], peak)
                        row["pbar_at_Q_half"] = value_at(tau, signals["pbar"], half)
                        row["T_at_Q_peak"] = value_at(tau, signals["T"], peak)
                        row["T_at_Q_half"] = value_at(tau, signals["T"], half)
                    event_rows.append(row)

        for boundary in ANALYSIS_BOUNDARIES:
            for m in orders[1:]:
                base = {name: computed[(1, boundary, name)] for name in ("Q", "T")}
                target = {name: computed[(m, boundary, name)] for name in ("Q", "T")}
                for gamma in GAMMAS:
                    hq1 = threshold(base["Q"], gamma)
                    hqm = threshold(target["Q"], gamma)
                    ht1 = threshold(base["T"], gamma)
                    htm = threshold(target["T"], gamma)
                    d1 = difference(hq1[0], ht1[0])
                    dm = difference(hqm[0], htm[0])
                    delta_q = difference(hq1[0], hqm[0])
                    delta_t = difference(ht1[0], htm[0])
                    residual = difference(delta_q, delta_t)
                    q_pair_status = paired_status(hq1, hqm)
                    t_pair_status = paired_status(ht1, htm)
                    qt_comparison_status = paired_status(hq1, hqm, ht1, htm)
                    q_root_jump_count = sum(
                        int(record.get("root_jumps", 0) or 0)
                        for record in (base["Q"], target["Q"])
                    )
                    t_root_jump_count = sum(
                        int(record.get("root_jumps", 0) or 0)
                        for record in (base["T"], target["T"])
                    )
                    if boundary == "actual_root":
                        q_root_branch_status = "root_branch_breaks" if q_root_jump_count else "continuous"
                        t_root_branch_status = "root_branch_breaks" if t_root_jump_count else "continuous"
                    else:
                        q_root_branch_status = "not_applicable"
                        t_root_branch_status = "not_applicable"
                    pair_rows.append({
                        "R0": R0,
                        "eta": eta,
                        "m": m,
                        "boundary": boundary,
                        "gamma": gamma,
                        "h_Q_ref": hq1[0],
                        "h_Q_m": hqm[0],
                        "h_T_ref": ht1[0],
                        "h_T_m": htm[0],
                        "d_ref_hQ_minus_hT": d1,
                        "d_m_hQ_minus_hT": dm,
                        "Delta_Q": delta_q,
                        "Delta_T": delta_t,
                        "E_DeltaQ_minus_DeltaT": residual,
                        "abs_E": None if residual is None else abs(residual),
                        "relative_E_over_DeltaQ": None if residual is None or delta_q is None or abs(delta_q) < 1e-8 else residual / delta_q,
                        # Keep event_status as a legacy alias for the joint Q/T
                        # comparison.  Q-only plots and tables must use
                        # Q_pair_status and the separate Q root-branch status.
                        "event_status": qt_comparison_status,
                        "Q_pair_status": q_pair_status,
                        "T_pair_status": t_pair_status,
                        "QT_comparison_status": qt_comparison_status,
                        "Q_root_branch_status": q_root_branch_status,
                        "T_root_branch_status": t_root_branch_status,
                        "Q_root_jump_count": q_root_jump_count,
                        "T_root_jump_count": t_root_jump_count,
                        "root_jump_count_across_QT_pair": q_root_jump_count + t_root_jump_count,
                        "Q_ref_status": hq1[1],
                        "Q_m_status": hqm[1],
                        "T_ref_status": ht1[1],
                        "T_m_status": htm[1],
                    })

        # Peak-independent decomposition on the proxy boundary, with one common tau*=0.
        boundary = "scaled_1"
        i_star = int(np.argmin(np.abs(tau)))
        if abs(tau[i_star]) > 1e-12:
            raise RuntimeError(f"tau=0 missing for R0={R0}, eta={eta}")
        for m in orders[1:]:
            arrays = {}
            valid = np.ones(len(tau), dtype=bool)
            for name in ("Q", "T", "pbar"):
                ref = np.asarray(curves[f"m1__{boundary}__{name}"], dtype=float)
                target = np.asarray(curves[f"m{m}__{boundary}__{name}"], dtype=float)
                floor = 1e-8 * max(float(np.nanmax(ref)), float(np.nanmax(target)))
                valid &= np.isfinite(ref) & np.isfinite(target) & (ref > floor) & (target > floor)
                if ref[i_star] <= floor or target[i_star] <= floor:
                    valid[:] = False
                arrays[name] = R0 * np.log(target * ref[i_star] / (ref * target[i_star]))
            common = valid & np.isfinite(arrays["Q"]) & np.isfinite(arrays["T"]) & np.isfinite(arrays["pbar"])
            identity_error = arrays["Q"] - arrays["T"] - arrays["pbar"]
            for i in np.flatnonzero(common):
                ratio_rows.append({
                    "R0": R0,
                    "eta": eta,
                    "m": m,
                    "tau": float(tau[i]),
                    "D_Q": float(arrays["Q"][i]),
                    "D_T": float(arrays["T"][i]),
                    "D_pbar": float(arrays["pbar"][i]),
                    "decomposition_error": float(identity_error[i]),
                })
            focus = common & (tau >= -0.5) & (tau <= 0.5)
            if np.count_nonzero(focus) >= 5:
                metrics = {}
                for name in ("Q", "T", "pbar"):
                    spline = CubicSpline(tau[focus], arrays[name][focus])
                    metrics[f"slope0_D_{name}"] = float(spline(0.0, 1))
                    metrics[f"rms_D_{name}"] = float(np.sqrt(np.mean(arrays[name][focus] ** 2)))
                    metrics[f"span_D_{name}"] = float(np.ptp(arrays[name][focus]))
                ratio_summary.append({
                    "R0": R0,
                    "eta": eta,
                    "m": m,
                    "tau_star": 0.0,
                    "window": "[-0.5,0.5]",
                    "valid_samples": int(np.count_nonzero(focus)),
                    "max_abs_decomposition_error": float(np.max(np.abs(identity_error[focus]))),
                    **metrics,
                })

        if R0 in ORIGINAL_RADII and eta in (0.35, 0.7):
            case_rows = [row for row in pair_rows if row["R0"] == R0 and row["eta"] == eta and row["gamma"] == 0.5]
            for m in (2, 3, 4):
                by_boundary = {row["boundary"]: row for row in case_rows if row["m"] == m}
                reference = by_boundary["scaled_1"]["Delta_Q"]
                wide = {"R0": R0, "eta": eta, "m": m}
                for boundary in BOUNDARIES:
                    row = by_boundary[boundary]
                    value = row["Delta_Q"]
                    short = {"scaled_0.9": "b0_0p9", "scaled_1": "b0", "scaled_1.1": "b0_1p1", "actual_root": "actual"}[boundary]
                    wide[f"Delta_{short}"] = value
                    wide[f"abs_change_from_b0_{short}"] = None if value is None or reference is None else abs(value - reference)
                    wide[f"signed_relative_change_from_b0_{short}"] = None if value is None or reference is None or abs(reference) < 1e-8 else (value - reference) / reference
                    q_status = row["Q_pair_status"]
                    if (boundary == "actual_root" and q_status == "defined" and
                            row["Q_root_branch_status"] == "root_branch_breaks"):
                        q_status = "defined_with_root_branch_breaks"
                    wide[f"status_{short}"] = q_status
                    wide[f"Q_pair_status_{short}"] = row["Q_pair_status"]
                    wide[f"Q_root_branch_status_{short}"] = row["Q_root_branch_status"]
                    wide[f"h_ref_{short}"] = row["h_Q_ref"]
                    wide[f"h_m_{short}"] = row["h_Q_m"]
                wide["small_b0_denominator"] = reference is None or abs(reference) < 1e-3
                boundary_wide.append(wide)

    write_csv("qt_signal_events.csv", event_rows)
    write_csv("qt_paired_events.csv", pair_rows)
    write_csv("qt_double_ratio_curves.csv", ratio_rows)
    write_csv("qt_double_ratio_summary.csv", ratio_summary)
    write_csv("four_boundary_original_radii.csv", boundary_wide)
    representative = [row for row in boundary_wide if row["R0"] in (24, 96)]
    write_csv("four_boundary_representative.csv", representative)
    stress = [row for row in pair_rows if row["R0"] in (24, 96) and row["eta"] in (0.35, 0.7) and row["m"] in (2, 3, 4) and row["gamma"] == 0.5 and row["boundary"] in ("scaled_0.8", "scaled_1", "scaled_1.2")]
    write_csv("boundary_20pct_stress_representative.csv", stress)

    defined_half = [row for row in pair_rows if row["gamma"] == 0.5 and row["boundary"] == "scaled_1" and row["QT_comparison_status"] == "defined"]
    residuals = np.array([row["E_DeltaQ_minus_DeltaT"] for row in defined_half], dtype=float)
    deltas = np.array([row["Delta_Q"] for row in defined_half], dtype=float)
    summary = {
        "source": "data/principal_refined; retained complex-field propagation, postprocessing only",
        "cases": len(cases),
        "beams": sum(len(meta["events"]) for meta, _ in cases),
        "qt_event_rows": len(event_rows),
        "paired_rows": len(pair_rows),
        "defined_proxy_half_pairs": len(defined_half),
        "undefined_proxy_half_pairs": len([row for row in pair_rows if row["gamma"] == 0.5 and row["boundary"] == "scaled_1" and row["QT_comparison_status"] != "defined"]),
        "actual_root_Q_defined_but_QT_unavailable": len([
            row for row in pair_rows
            if row["gamma"] == 0.5 and row["boundary"] == "actual_root"
            and row["Q_pair_status"] == "defined"
            and row["QT_comparison_status"] != "defined"
        ]),
        "max_abs_Q_minus_Tpbar": identity_max,
        "median_abs_E_proxy_half": float(np.median(np.abs(residuals))),
        "max_abs_E_proxy_half": float(np.max(np.abs(residuals))),
        "median_abs_E_over_abs_DeltaQ_proxy_half": float(np.median(np.abs(residuals) / np.maximum(np.abs(deltas), 1e-15))),
        "representative_R24_eta0p7_m4": next(row for row in defined_half if row["R0"] == 24 and row["eta"] == 0.7 and row["m"] == 4),
        "representative_R96_eta0p7_m4": next(row for row in defined_half if row["R0"] == 96 and row["eta"] == 0.7 and row["m"] == 4),
        "actual_root_R16_eta0p7_m4": next(
            row for row in pair_rows
            if row["R0"] == 16 and row["eta"] == 0.7 and row["m"] == 4
            and row["gamma"] == 0.5 and row["boundary"] == "actual_root"
        ),
    }
    (DEST / "qt_boundary_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
