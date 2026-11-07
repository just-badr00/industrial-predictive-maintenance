import pytest

from src.modeling.predict import health_status


@pytest.mark.parametrize(
    "predicted_rul, expected_status",
    [
        (0.0, "critical"),
        (10.0, "critical"),
        (10.01, "warning"),
        (30.0, "warning"),
        (30.01, "healthy"),
    ],
)
def test_health_status_boundaries(predicted_rul, expected_status):
    assert health_status(predicted_rul) == expected_status
