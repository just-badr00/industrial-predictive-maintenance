from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.modeling.explain import explain_predictions


def test_factor_ranking_preserves_signs_values_and_row_alignment():
    # Nonzero labels catch accidental use of label-based .loc for row positions.
    inputs = pd.DataFrame([[11.0, 22.0, 33.0, 44.0], [55.0, 66.0, 77.0, 88.0]],
                          columns=["a", "b", "c", "d"], index=[30, 48])

    def fake_explainer(frame, check_additivity):
        assert frame is inputs
        assert check_additivity is True
        return SimpleNamespace(
            values=np.array([[-20.0, 5.0, 12.0, 1.0], [1.0, -9.0, 3.0, 15.0]]),
            feature_names=inputs.columns.tolist(),
        )

    factors = explain_predictions(fake_explainer, inputs)
    assert factors == [
        [{"feature": "a", "value": 11.0, "shap_value": -20.0},
         {"feature": "c", "value": 33.0, "shap_value": 12.0},
         {"feature": "b", "value": 22.0, "shap_value": 5.0}],
        [{"feature": "d", "value": 88.0, "shap_value": 15.0},
         {"feature": "b", "value": 66.0, "shap_value": -9.0},
         {"feature": "c", "value": 77.0, "shap_value": 3.0}],
    ]
