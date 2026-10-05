"""Generate Q and T event-difference coefficients without propagation data."""
from pathlib import Path
import csv

import numpy as np

from formal_event_coefficients import airy_reference_event, coefficients


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "focused_refinement" / "qt_formal_coefficients.csv"


rows = []
for eta in (0.35, 0.7, 0.9):
    airy_event = airy_reference_event(eta, 0.5)
    for observable in ("Q", "T"):
        reference = coefficients(1, eta, points=25, nx=1201, event_record=airy_event, observable=observable)
        reference_fine = coefficients(1, eta, points=29, nx=1601, event_record=airy_event, observable=observable)
        for m in (2, 3, 4):
            target = coefficients(m, eta, points=25, nx=1201, event_record=airy_event, observable=observable)
            target_fine = coefficients(m, eta, points=29, nx=1601, event_record=airy_event, observable=observable)
            difference = np.asarray(reference["half_height_coefficients"]) - np.asarray(target["half_height_coefficients"])
            difference_fine = np.asarray(reference_fine["half_height_coefficients"]) - np.asarray(target_fine["half_height_coefficients"])
            rows.append({
                "eta": eta,
                "m": m,
                "observable": observable,
                "epsilon_coefficient": difference[1],
                "epsilon2_coefficient": difference[2],
                "epsilon3_coefficient": difference[3],
                "epsilon4_coefficient": difference[4],
                "max_refinement_change": float(np.max(np.abs(difference_fine - difference))),
                "propagation_fit_used": False,
            })

with DEST.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(DEST)
