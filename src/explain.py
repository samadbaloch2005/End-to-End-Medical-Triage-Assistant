"""Per-patient explanations for the triage model.

The served model is a calibrated pipeline (preprocessing, SMOTE, classifier and
Platt scaling), so tree-specific tools like TreeSHAP can't be applied to it
directly. We use a small model-agnostic method instead, measured against a
healthy reference patient:

    remove: start from the patient and reset one feature to the reference value
    add:    start from the reference and give it the patient's value for one feature

Each feature's contribution is the average of the two effects. Looking only at
"remove" under-credits features when a patient is already clearly critical
(one bad vital is enough, so removing any single one changes little); "add"
has the opposite bias. Averaging both is a cheap approximation of Shapley values.

Effects are measured on expected risk, P(Medium) * 1 + P(High) * 2, a number
from 0 to 2. Positive means "this value pushes the patient to a higher tier".
It needs 2 extra predictions per feature, which is fast enough to run on every
triage, and it explains the exact model that made the decision.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.preprocess import FEATURES


def expected_risk(proba: np.ndarray) -> np.ndarray:
    return proba[:, 1] * 1.0 + proba[:, 2] * 2.0


def explain(model: Any, patient: dict, reference: dict) -> list[dict]:
    """Return feature contributions sorted by absolute impact, largest first."""
    base_row = {f: patient.get(f) for f in FEATURES}
    ref_row = {f: reference[f] for f in FEATURES}
    rows = [base_row, ref_row]
    for feature in FEATURES:
        rows.append({**base_row, feature: ref_row[feature]})
        rows.append({**ref_row, feature: base_row[feature]})

    risk = expected_risk(model.predict_proba(pd.DataFrame(rows, columns=FEATURES)))
    actual, reference_risk = risk[0], risk[1]

    contributions = []
    for i, feature in enumerate(FEATURES):
        if base_row[feature] is None:
            continue
        removed = actual - risk[2 + 2 * i]
        added = risk[3 + 2 * i] - reference_risk
        contributions.append(
            {"feature": feature, "value": base_row[feature], "impact": float((removed + added) / 2)}
        )
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions
