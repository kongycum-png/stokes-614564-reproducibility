"""Generate non-fitted A, B, C event coefficients for one (m, eta)."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.special import jnp_zeros

from formal_event_coefficients import airy_reference_event, coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--reference-m", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--observable", choices=("Q", "T"), default="Q")
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--nx", type=int, default=1201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.m < 1 or args.reference_m < 1:
        raise ValueError("m and reference-m must be positive fixed integers")

    airy_event = airy_reference_event(args.eta, args.gamma)
    reference = coefficients(args.reference_m, args.eta, args.points, args.nx, args.gamma, airy_event, args.observable)
    target = coefficients(args.m, args.eta, args.points, args.nx, args.gamma, airy_event, args.observable)
    delta = np.asarray(reference["half_height_coefficients"]) - np.asarray(target["half_height_coefficients"])
    repeat_reference = coefficients(args.reference_m, args.eta, args.points + 4, args.nx + 400, args.gamma, airy_event, args.observable)
    repeat_target = coefficients(args.m, args.eta, args.points + 4, args.nx + 400, args.gamma, airy_event, args.observable)
    repeat_delta = np.asarray(repeat_reference["half_height_coefficients"]) - np.asarray(repeat_target["half_height_coefficients"])
    D_reference = args.reference_m**2 + 0.75 - float(jnp_zeros(args.reference_m, 1)[0]) ** 2
    D_target = args.m**2 + 0.75 - float(jnp_zeros(args.m, 1)[0]) ** 2
    K = args.gamma * (airy_event["peak"] - airy_event["crossing"]) * airy_event["peak_value"] / airy_event["crossing_slope"]
    result = {
        "status": "FORMAL_INWARD_BRANCH_COEFFICIENTS_NOT_FIT_TO_PROPAGATION",
        "m": args.m,
        "reference_m": args.reference_m,
        "eta": args.eta,
        "gamma": args.gamma,
        "observable": args.observable,
        "epsilon": "R0^(-1/2)",
        "event": airy_event,
        "reference_event_coefficients": reference,
        "target_event_coefficients": target,
        "advance_coefficients_epsilon_0_to_4": delta.tolist(),
        "event_difference_coefficients": {"epsilon": float(delta[1]), "epsilon2": float(delta[2]), "epsilon3": float(delta[3]), "epsilon4": float(delta[4])},
        "A": float(delta[2]) if args.observable == "Q" else None,
        "B": float(delta[3]) if args.observable == "Q" else None,
        "C": float(delta[4]) if args.observable == "Q" else None,
        "A_from_K_times_D_difference": float(K * (D_reference - D_target)) if args.observable == "Q" else None,
        "A_identity_error": float(delta[2] - K * (D_reference - D_target)) if args.observable == "Q" else None,
        "refinement_difference_epsilon_0_to_4": (repeat_delta - delta).tolist(),
        "retained_terms": [
            "complex Airy argument and derivatives",
            "full amplitude g",
            "radial Jacobian",
            "first required Debye term",
            "all field cross-products through consistent intensity order epsilon^4"
        ],
        "excluded_from_local_series": ["outward Hankel branch", "spectral endpoint"],
        "reads_propagation_fit_data": False,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
