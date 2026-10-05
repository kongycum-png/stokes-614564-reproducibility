"""Focused count-level diagnosis of the retained measurement failure.

The existing baseline results are not overwritten.  The frozen protocol adds
m=3, separates ideal/detector/estimator/random stages, and compares the
retained blind estimator with a known-peak-location-assisted diagnostic and
one pre-frozen, wider-smoothing blind estimator.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from focused_measurement_estimator import estimate
from measurement_detector import detector_response, shifted_means
from observables import event


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "focused_measurement"
DEST.mkdir(exist_ok=True)
PROTOCOL_PATH = ROOT / "config" / "MEASUREMENT_FOCUSED_PROTOCOL.json"
BASELINE_PATH = ROOT / "config" / "SYNTHETIC_MEASUREMENT.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["baseline"]
ORDERS = PROTOCOL["orders"]
ESTIMATORS = {
    "known_peak_location_assisted": (0.25, "nearest_to_reference_peak"),
    "current_blind": (0.25, "largest"),
    "wide_blind": (0.75, "largest"),
}


def write_csv(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def half(record: dict) -> float | None:
    return record.get("thresholds", {}).get("0.5", {}).get("tau")


def interp(tau: np.ndarray, values: np.ndarray, x: float) -> float:
    return float(np.interp(x, tau, values))


def noiseless_estimates(tau: np.ndarray, q_detector: np.ndarray, detector_peaks: np.ndarray) -> dict:
    result = {}
    for name, (width, selection) in ESTIMATORS.items():
        halves, peaks, indices = [], [], []
        for j in range(len(ORDERS)):
            est = estimate(
                tau,
                q_detector[j],
                smoothing_width_tau=width,
                selection=selection,
                reference_peak_tau=detector_peaks[j] if selection == "nearest_to_reference_peak" else None,
            )
            halves.append(float(est["half_tau"][0]))
            peaks.append(float(est["peak_tau"][0]))
            indices.append(float(est["selected_index"][0]))
        result[name] = {"half_tau": np.array(halves), "peak_tau": np.array(peaks), "selected_index": np.array(indices)}
    return result


def simulate(response: dict, cfg: dict, nrep: int, seed: int, detector_peaks: np.ndarray) -> dict:
    rng = np.random.default_rng(seed)
    mu = response["mu"]
    tau = response["tau"]
    n = len(tau)
    N = cfg["Nincident"]
    npixels = response["npixels"]
    c = cfg["polarimetric_cross_talk_mean"]
    rho = cfg["gain_cross_order_correlation"]
    shape = (tau - (tau[0] + tau[-1]) / 2) / ((tau[-1] - tau[0]) / 2)
    store = {
        name: {
            "half_mm": [],
            "peak_mm": [],
            "selected_index": [],
            "n_local_maxima": [],
        }
        for name in ESTIMATORS
    }
    raw_example = None

    for start in range(0, nrep, 100):
        nr = min(100, nrep - start)
        gain = cfg.get("gain_mean", 0.0) + cfg["gain_sd"] * (
            np.sqrt(rho) * rng.normal(size=(nr, 1))
            + np.sqrt(1 - rho) * rng.normal(size=(nr, len(ORDERS)))
        )
        asymmetry = rng.normal(0, cfg["polarimetric_cross_talk_asymmetry_sd"], size=(nr, 1, 1))
        mix_a = c + asymmetry / 2
        mix_b = c - asymmetry / 2
        offset = (
            rng.normal(0, cfg["common_axial_offset_sd_mm"], size=(nr, 1, 1))
            + rng.normal(0, cfg["order_axial_offset_sd_mm"], size=(nr, len(ORDERS), 1))
            + rng.normal(0, cfg["smooth_scan_drift_sd_mm"], size=(nr, len(ORDERS), 1)) * shape[None, None, :]
        )
        shifted = shifted_means(response, offset)
        plus = ((1 - mix_a) * shifted[:, :, 0] + mix_b * shifted[:, :, 1]) * (1 + gain[:, :, None])
        minus = (mix_a * shifted[:, :, 0] + (1 - mix_b) * shifted[:, :, 1]) * (1 - gain[:, :, None])
        optical = np.stack([plus, minus], axis=2)
        expected = N * optical + cfg["background_e_per_pixel"] * npixels[None, :, None, :]
        counts = rng.poisson(expected) + rng.normal(size=expected.shape) * np.sqrt(npixels)[None, :, None, :] * cfg["read_noise_e_per_pixel"]
        background_rho = cfg["background_cross_channel_correlation"]
        background_error = cfg["background_calibration_sd_e_per_pixel"] * (
            np.sqrt(background_rho) * rng.normal(size=(nr, 1))
            + np.sqrt(1 - background_rho) * rng.normal(size=(nr, 2))
        )
        corrected = counts - (cfg["background_e_per_pixel"] + background_error[:, None, :, None]) * npixels[None, :, None, :]
        monitor = rng.poisson(N * cfg["monitor_reference_count_multiplier"], size=(nr, n)) / cfg["monitor_reference_count_multiplier"]
        signal = (corrected[:, :, 0] - corrected[:, :, 1]) / monitor[:, None, :]
        if raw_example is None:
            raw_example = {
                "raw_aggregate_Iplus_Iminus": counts[:3],
                "expected_raw_aggregate": expected[:3],
                "reference_monitor": monitor[:3],
            }

        for name, (width, selection) in ESTIMATORS.items():
            halves, peaks, indices, maxima = [], [], [], []
            for j in range(len(ORDERS)):
                est = estimate(
                    tau,
                    signal[:, j],
                    smoothing_width_tau=width,
                    selection=selection,
                    reference_peak_tau=detector_peaks[j] if selection == "nearest_to_reference_peak" else None,
                )
                halves.append(est["half_tau"] * response["Lc"])
                peaks.append(est["peak_tau"] * response["Lc"])
                indices.append(est["selected_index"])
                maxima.append(est["n_local_maxima"])
            store[name]["half_mm"].append(np.array(halves).T)
            store[name]["peak_mm"].append(np.array(peaks).T)
            store[name]["selected_index"].append(np.array(indices).T)
            store[name]["n_local_maxima"].append(np.array(maxima).T)

    for name in ESTIMATORS:
        for key in store[name]:
            store[name][key] = np.concatenate(store[name][key])
    return {"estimates": store, "raw_example": raw_example}


def covariance_pairwise(values: np.ndarray) -> np.ndarray:
    out = np.full((values.shape[1], values.shape[1]), np.nan)
    for i in range(values.shape[1]):
        for j in range(values.shape[1]):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if valid.sum() > 2:
                out[i, j] = np.cov(values[valid, i], values[valid, j], ddof=1)[0, 1]
    return out


def summarize_group(group_index: int, group: dict) -> tuple[list[dict], list[dict], dict]:
    R0 = int(group["R0"])
    eta = float(group["eta"])
    cfg = dict(BASELINE, Nincident=float(group["Nincident"]))
    tag = f"R{R0}_eta{eta:g}_N{cfg['Nincident']:.0e}"
    fields = dict(np.load(ROOT / "data" / "principal_refined" / f"R{R0}_eta{eta:g}_fields.npz"))
    meta = json.loads((ROOT / "data" / "principal_refined" / f"R{R0}_eta{eta:g}.json").read_text(encoding="utf-8"))
    Lc = float(fields["kw"]) * float(fields["w_mm"]) / np.sqrt(R0)
    scan_min, scan_max = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["scan_tau"]
    tau = scan_min + np.arange(int(np.floor((scan_max - scan_min) * Lc / cfg["dz_mm"])) + 1) * cfg["dz_mm"] / Lc
    response = detector_response(fields, ORDERS, tau, cfg["pixel_pitch_mm"], cfg["blur_sigma_mm"])
    mu = response["mu"]
    c = cfg["polarimetric_cross_talk_mean"]
    gain_mean = cfg.get("gain_mean", 0.0)
    q_detector = (1 - 2 * c) * (mu[:, 0] - mu[:, 1]) + gain_mean * mu.sum(axis=1)

    ideal_half_tau = np.array([half(meta["events"][str(m)]["scaled_1"]) for m in ORDERS], dtype=float)
    ideal_peak_tau = np.array([meta["events"][str(m)]["scaled_1"]["tau_peak"] for m in ORDERS], dtype=float)
    detector_events = [event(tau, q_detector[j]) for j in range(len(ORDERS))]
    detector_half_tau = np.array([half(record) for record in detector_events], dtype=float)
    detector_peak_tau = np.array([record["tau_peak"] for record in detector_events], dtype=float)
    no_noise = noiseless_estimates(tau, q_detector, detector_peak_tau)

    ncal = int(PROTOCOL["calibration_replicates"])
    nval = int(PROTOCOL["validation_replicates"])
    seed = int(PROTOCOL["seed_base"]) + 20 * group_index
    calibration = simulate(response, cfg, ncal, seed, detector_peak_tau)
    validation = simulate(response, cfg, nval, seed + 1, detector_peak_tau)

    ideal_mm = ideal_half_tau * Lc
    detector_mm = detector_half_tau * Lc
    event_rows = []
    pair_rows = []
    save = {"tau": tau, "ideal_half_mm": ideal_mm, "detector_half_mm": detector_mm, "ideal_peak_mm": ideal_peak_tau * Lc, "detector_peak_mm": detector_peak_tau * Lc}

    for name in ESTIMATORS:
        cal = calibration["estimates"][name]
        val = validation["estimates"][name]
        covariance = covariance_pairwise(cal["half_mm"])
        save[f"{name}_cal_half_mm"] = cal["half_mm"]
        save[f"{name}_val_half_mm"] = val["half_mm"]
        save[f"{name}_val_peak_mm"] = val["peak_mm"]
        save[f"{name}_val_selected_index"] = val["selected_index"]
        save[f"{name}_val_n_local_maxima"] = val["n_local_maxima"]

        for j, m in enumerate(ORDERS):
            values = val["half_mm"][:, j]
            valid = np.isfinite(values)
            optical_counts = cfg["Nincident"] * mu[j]
            row = {
                "tag": tag,
                "R0": R0,
                "eta": eta,
                "Nincident": cfg["Nincident"],
                "m": m,
                "estimator": name,
                "h_ideal_mm": ideal_mm[j],
                "h_detector_mm": detector_mm[j],
                "h_estimator0_mm": no_noise[name]["half_tau"][j] * Lc,
                "mean_h_random_mm": float(np.nanmean(values)),
                "imaging_bias_mm": detector_mm[j] - ideal_mm[j],
                "deterministic_estimator_bias_mm": no_noise[name]["half_tau"][j] * Lc - detector_mm[j],
                "random_bias_from_estimator0_mm": float(np.nanmean(values)) - no_noise[name]["half_tau"][j] * Lc,
                "validation_extraction_rate": float(valid.mean()),
                "validation_replicates": nval,
                "calibrated_event_sd_mm": float(np.sqrt(covariance[j, j])),
                "mean_local_maxima": float(np.mean(val["n_local_maxima"][:, j])),
                "ROI_expected_Iplus_at_detector_half": interp(tau, optical_counts[0], detector_half_tau[j]),
                "ROI_expected_Iminus_at_detector_half": interp(tau, optical_counts[1], detector_half_tau[j]),
                "ROI_expected_Iplus_at_detector_peak": interp(tau, optical_counts[0], detector_peak_tau[j]),
                "ROI_expected_Iminus_at_detector_peak": interp(tau, optical_counts[1], detector_peak_tau[j]),
                "ROI_expected_total_count_range": f"[{optical_counts.sum(axis=0).min():.6g},{optical_counts.sum(axis=0).max():.6g}]",
            }
            if name == "current_blind":
                assisted_index = validation["estimates"]["known_peak_location_assisted"]["selected_index"][:, j]
                both = np.isfinite(val["selected_index"][:, j]) & np.isfinite(assisted_index)
                same = both & (val["selected_index"][:, j] == assisted_index)
                row["same_discrete_local_maximum_fraction_all"] = float(same.mean())
                row["same_discrete_local_maximum_fraction_conditional"] = float(same.sum() / both.sum()) if both.any() else None
            event_rows.append(row)

        for j, m in enumerate(ORDERS[1:], start=1):
            cal_delta = cal["half_mm"][:, 0] - cal["half_mm"][:, j]
            val_delta = val["half_mm"][:, 0] - val["half_mm"][:, j]
            cal_valid = np.isfinite(cal_delta)
            val_valid = np.isfinite(val_delta)
            calibrated_sd = float(np.std(cal_delta[cal_valid], ddof=1))
            half_width = 1.95996398454 * calibrated_sd
            delta_ideal = ideal_mm[0] - ideal_mm[j]
            delta_detector = detector_mm[0] - detector_mm[j]
            delta_estimator0 = (no_noise[name]["half_tau"][0] - no_noise[name]["half_tau"][j]) * Lc
            sign_good = val_valid & (np.sign(val_delta) == np.sign(delta_ideal))
            cover_ideal = val_valid & (np.abs(val_delta - delta_ideal) <= half_width)
            cover_detector = val_valid & (np.abs(val_delta - delta_detector) <= half_width)
            pair_rows.append({
                "tag": tag,
                "R0": R0,
                "eta": eta,
                "Nincident": cfg["Nincident"],
                "m": m,
                "estimator": name,
                "Delta_ideal_mm": delta_ideal,
                "Delta_detector_mm": delta_detector,
                "Delta_estimator0_mm": delta_estimator0,
                "Delta_random_mean_mm": float(np.nanmean(val_delta)),
                "imaging_bias_mm": delta_detector - delta_ideal,
                "deterministic_estimator_bias_mm": delta_estimator0 - delta_detector,
                "random_bias_from_estimator0_mm": float(np.nanmean(val_delta)) - delta_estimator0,
                "total_bias_vs_ideal_mm": float(np.nanmean(val_delta)) - delta_ideal,
                "calibration_pair_extraction_rate": float(cal_valid.mean()),
                "validation_pair_extraction_rate": float(val_valid.mean()),
                "calibrated_delta_sd_mm": calibrated_sd,
                "calibrated_95_CI_width_mm": 2 * half_width,
                "coverage_ideal_all": float(cover_ideal.mean()),
                "coverage_ideal_conditional": float(cover_ideal.sum() / val_valid.sum()) if val_valid.any() else None,
                "coverage_detector_all": float(cover_detector.mean()),
                "coverage_detector_conditional": float(cover_detector.sum() / val_valid.sum()) if val_valid.any() else None,
                "sign_recovery_all": float(sign_good.mean()),
                "sign_recovery_conditional": float(sign_good.sum() / val_valid.sum()) if val_valid.any() else None,
                "validation_extracted_pairs": int(val_valid.sum()),
                "validation_replicates": nval,
                "calibration_cross_order_covariance_mm2": covariance[0, j],
            })

    for key, value in calibration["raw_example"].items():
        save[f"calibration_{key}"] = value
    np.savez_compressed(DEST / f"{tag}_focused_samples.npz", **save)
    group_record = {
        "tag": tag,
        "config": cfg,
        "Lc_mm": Lc,
        "orders": ORDERS,
        "ideal_half_tau": ideal_half_tau.tolist(),
        "detector_half_tau": detector_half_tau.tolist(),
        "detector_peak_tau": detector_peak_tau.tolist(),
        "noiseless_estimators": {name: {key: value.tolist() for key, value in record.items()} for name, record in no_noise.items()},
        "calibration_seed": seed,
        "validation_seed": seed + 1,
        "calibration_replicates": ncal,
        "validation_replicates": nval,
    }
    (DEST / f"{tag}.json").write_text(json.dumps(group_record, indent=2) + "\n", encoding="utf-8")
    return event_rows, pair_rows, group_record


def main() -> None:
    all_events, all_pairs, groups = [], [], []
    for index, group in enumerate(PROTOCOL["groups"]):
        print("START", group, flush=True)
        events, pairs, record = summarize_group(index, group)
        all_events.extend(events)
        all_pairs.extend(pairs)
        groups.append(record)
        print("END", record["tag"], flush=True)
    write_csv("measurement_stage_events.csv", all_events)
    write_csv("measurement_stage_pairs.csv", all_pairs)
    report = {
        "status": "RUN_COMPLETED",
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": [record["tag"] for record in groups],
        "event_rows": len(all_events),
        "pair_rows": len(all_pairs),
        "channel_definition": "Each circular channel intensity includes the 1/2 factor in measurement_detector.detector_response; Poisson draws are made from separate I+ and I- aggregate counts before differencing.",
        "assisted_diagnostic_role": "Uses the noiseless detector peak location to select the nearest noisy local maximum. It does not fix the noisy peak value, physical branch, or half-height crossing and is excluded from experimental-feasibility claims.",
    }
    (DEST / "measurement_focused_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
