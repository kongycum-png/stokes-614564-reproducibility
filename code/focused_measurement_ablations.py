"""Targeted ablations for the R0=96, eta=0.7, N=1e6 failure."""
from pathlib import Path
import csv
import json

import numpy as np

from focused_measurement_diagnostics import BASELINE, ESTIMATORS, ORDERS, noiseless_estimates, simulate
from measurement_detector import detector_response
from observables import event


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "focused_measurement"
P = json.loads((ROOT / "config" / "MEASUREMENT_FOCUSED_ABLATION_PROTOCOL.json").read_text())
SOURCE_CONFIG = json.loads((ROOT / "config" / "SYNTHETIC_MEASUREMENT.json").read_text())


def half(record):
    return record["thresholds"]["0.5"]["tau"]


case = P["case"]
R0, eta = int(case["R0"]), float(case["eta"])
fields = dict(np.load(ROOT / "data" / "principal_refined" / f"R{R0}_eta{eta:g}_fields.npz"))
meta = json.loads((ROOT / "data" / "principal_refined" / f"R{R0}_eta{eta:g}.json").read_text())
Lc = float(fields["kw"]) * float(fields["w_mm"]) / np.sqrt(R0)
scan_min, scan_max = SOURCE_CONFIG["scan_tau"]
rows = []

for scenario_index, (scenario, changes) in enumerate(P["scenarios"].items()):
    cfg = dict(BASELINE, Nincident=float(case["Nincident"]), **changes)
    tau = scan_min + np.arange(int(np.floor((scan_max - scan_min) * Lc / cfg["dz_mm"])) + 1) * cfg["dz_mm"] / Lc
    response = detector_response(fields, ORDERS, tau, cfg["pixel_pitch_mm"], cfg["blur_sigma_mm"])
    mu = response["mu"]
    c = cfg["polarimetric_cross_talk_mean"]
    q_detector = (1 - 2 * c) * (mu[:, 0] - mu[:, 1]) + cfg.get("gain_mean", 0.0) * mu.sum(axis=1)
    detector_events = [event(tau, q_detector[j]) for j in range(len(ORDERS))]
    detector_peaks = np.array([record["tau_peak"] for record in detector_events])
    no_noise = noiseless_estimates(tau, q_detector, detector_peaks)
    ideal = np.array([half(meta["events"][str(m)]["scaled_1"]) for m in ORDERS]) * Lc
    detector_half = np.array([half(record) for record in detector_events]) * Lc
    seed = int(P["seed_base"]) + 20 * scenario_index
    calibration = simulate(response, cfg, int(P["calibration_replicates"]), seed, detector_peaks)["estimates"]
    validation = simulate(response, cfg, int(P["validation_replicates"]), seed + 1, detector_peaks)["estimates"]
    for estimator in ESTIMATORS:
        for j, m in enumerate(ORDERS[1:], start=1):
            cal_delta = calibration[estimator]["half_mm"][:, 0] - calibration[estimator]["half_mm"][:, j]
            val_delta = validation[estimator]["half_mm"][:, 0] - validation[estimator]["half_mm"][:, j]
            cvalid = np.isfinite(cal_delta)
            vvalid = np.isfinite(val_delta)
            target = ideal[0] - ideal[j]
            target_detector = detector_half[0] - detector_half[j]
            target_estimator0 = (no_noise[estimator]["half_tau"][0] - no_noise[estimator]["half_tau"][j]) * Lc
            sd = float(np.std(cal_delta[cvalid], ddof=1))
            halfwidth = 1.95996398454 * sd
            sign = vvalid & (np.sign(val_delta) == np.sign(target))
            cover = vvalid & (np.abs(val_delta - target) <= halfwidth)
            rows.append({
                "scenario": scenario,
                "m": m,
                "estimator": estimator,
                "Delta_ideal_mm": target,
                "Delta_detector_mm": target_detector,
                "Delta_estimator0_mm": target_estimator0,
                "Delta_random_mean_mm": float(np.nanmean(val_delta)),
                "total_bias_mm": float(np.nanmean(val_delta)) - target,
                "validation_extraction_rate": float(vvalid.mean()),
                "sign_recovery_all": float(sign.mean()),
                "sign_recovery_conditional": float(sign.sum() / vvalid.sum()),
                "calibrated_95_CI_width_mm": 2 * halfwidth,
                "coverage_ideal_all": float(cover.mean()),
                "coverage_ideal_conditional": float(cover.sum() / vvalid.sum()),
            })

with (DEST / "measurement_failure_ablations.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"status": "RUN_COMPLETED", "rows": len(rows), "scenarios": list(P["scenarios"])}, indent=2))
