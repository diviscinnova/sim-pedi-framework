# Tests for SIM-PEDI calculations and input validation.

import pandas as pd
import pytest

from computation.clearance import (
    calculate_size_factor,
    calculate_size_maturation_clearance,
    calculate_weight_only_clearance,
)
from computation.main import load_profiles, run_analysis
from computation.maturation import (
    calculate_maturation_factor,
    calculate_pma_weeks,
)


@pytest.fixture
def profiles():
    return pd.DataFrame(
        {
            "profile_id": ["P001", "P002"],
            "postnatal_age_days": [14, 14],
            "gestational_age_at_birth_weeks": [32, 40],
            "weight_kg": [3.0, 3.0],
            "sex": ["female", "male"],
        }
    )


def test_pma_uses_individual_gestational_age():
    assert calculate_pma_weeks(14, 32) == pytest.approx(34)
    assert calculate_pma_weeks(14, 40) == pytest.approx(42)


def test_pma_accepts_birth_day():
    assert calculate_pma_weeks(0, 36) == pytest.approx(36)


@pytest.mark.parametrize(
    "postnatal_age,gestational_age",
    [
        (-1, 40),
        (14, 0),
        (14, -1),
        (float("nan"), 40),
        (14, float("inf")),
    ],
)
def test_pma_rejects_invalid_inputs(postnatal_age, gestational_age):
    with pytest.raises(ValueError):
        calculate_pma_weeks(postnatal_age, gestational_age)


def test_maturation_equals_half_at_pma50():
    assert calculate_maturation_factor(47.7) == pytest.approx(0.5)


def test_maturation_matches_direct_equation():
    pma = 60.0
    pma50 = 47.7
    hill = 3.4

    expected = pma**hill / (pma**hill + pma50**hill)

    assert calculate_maturation_factor(
        pma, pma50, hill
    ) == pytest.approx(expected)


def test_maturation_increases_with_pma():
    factors = [
        calculate_maturation_factor(pma)
        for pma in (30, 40, 50, 100, 800)
    ]

    assert all(0 < factor < 1 for factor in factors)
    assert all(
        earlier < later
        for earlier, later in zip(factors, factors[1:])
    )


@pytest.mark.parametrize(
    "invalid",
    [0, -1, float("nan"), float("inf")],
)
@pytest.mark.parametrize(
    "parameter",
    ["pma_weeks", "pma50_weeks", "hill"],
)
def test_maturation_rejects_invalid_parameters(parameter, invalid):
    arguments = {
        "pma_weeks": 50.0,
        "pma50_weeks": 47.7,
        "hill": 3.4,
    }
    arguments[parameter] = invalid

    with pytest.raises(ValueError):
        calculate_maturation_factor(**arguments)


def test_size_factor_at_reference_weight():
    assert calculate_size_factor(70) == pytest.approx(1)


def test_weight_only_at_reference_weight():
    assert calculate_weight_only_clearance(
        70, adult_clearance=100
    ) == pytest.approx(100)


def test_weight_only_increases_with_weight():
    assert (
        calculate_weight_only_clearance(5)
        < calculate_weight_only_clearance(10)
    )


def test_combined_model_matches_product():
    expected = (
        calculate_weight_only_clearance(4)
        * calculate_maturation_factor(44)
    )

    actual = calculate_size_maturation_clearance(
        weight_kg=4,
        pma_weeks=44,
    )

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "invalid",
    [0, -1, float("nan"), float("inf")],
)
def test_clearance_rejects_invalid_weight(invalid):
    with pytest.raises(ValueError):
        calculate_weight_only_clearance(invalid)

    with pytest.raises(ValueError):
        calculate_size_maturation_clearance(invalid, 44)


@pytest.mark.parametrize(
    "invalid",
    [0, -1, float("nan"), float("inf")],
)
def test_both_models_reject_invalid_reference_clearance(invalid):
    with pytest.raises(ValueError):
        calculate_weight_only_clearance(
            4, adult_clearance=invalid
        )

    with pytest.raises(ValueError):
        calculate_size_maturation_clearance(
            4, 44, adult_clearance=invalid
        )


def test_analysis_calculates_consistent_outputs(profiles):
    results = run_analysis(profiles)

    assert results["pma_weeks"].tolist() == pytest.approx([34, 42])

    assert results["clearance_size_maturation"].to_numpy() == (
        pytest.approx(
            (
                results["clearance_weight_only"]
                * results["maturation_factor"]
            ).to_numpy()
        )
    )

    assert results["percentage_difference"].to_numpy() == (
        pytest.approx(
            (
                100 * (results["maturation_factor"] - 1)
            ).to_numpy()
        )
    )

    assert (
        results.loc[0, "clearance_size_maturation"]
        < results.loc[1, "clearance_size_maturation"]
    )


def test_load_profiles_accepts_valid_csv(tmp_path, profiles):
    path = tmp_path / "profiles.csv"
    profiles.to_csv(path, index=False)

    loaded = load_profiles(path)

    assert len(loaded) == 2


def test_load_profiles_rejects_missing_columns(tmp_path, profiles):
    path = tmp_path / "profiles.csv"
    profiles.drop(columns=["weight_kg"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_profiles(path)


def test_load_profiles_rejects_duplicate_ids(tmp_path, profiles):
    path = tmp_path / "profiles.csv"
    profiles["profile_id"] = ["P001", "P001"]
    profiles.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique"):
        load_profiles(path)


def test_load_profiles_rejects_missing_numbers(tmp_path, profiles):
    path = tmp_path / "profiles.csv"
    profiles.loc[0, "weight_kg"] = float("nan")
    profiles.to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite"):
        load_profiles(path)