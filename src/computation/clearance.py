# Illustrative size-only and size-plus-maturation clearance models.

from math import isfinite

from computation.maturation import (
    DEFAULT_HILL,
    DEFAULT_PMA50_WEEKS,
    calculate_maturation_factor,
)


ADULT_REFERENCE_WEIGHT_KG = 70.0
ALLOMETRIC_EXPONENT = 0.75
DEFAULT_ADULT_CLEARANCE = 100.0


def _validate_adult_clearance(adult_clearance: float) -> None:
    if not isfinite(adult_clearance) or adult_clearance <= 0:
        raise ValueError(
            "Reference clearance must be finite and positive."
        )


def calculate_size_factor(weight_kg: float) -> float:
    # Calculate the dimensionless allometric body-size factor.
    if not isfinite(weight_kg) or weight_kg <= 0:
        raise ValueError(
            "Body weight must be finite and positive."
        )

    return (
        weight_kg / ADULT_REFERENCE_WEIGHT_KG
    ) ** ALLOMETRIC_EXPONENT


def calculate_weight_only_clearance(
    weight_kg: float,
    adult_clearance: float = DEFAULT_ADULT_CLEARANCE,
) -> float:
    """
    Calculate Model A: weight-only allometric clearance.

    Output uses the same units as adult_clearance. The default
    reference of 100 represents arbitrary simulation units.
    """
    _validate_adult_clearance(adult_clearance)

    return adult_clearance * calculate_size_factor(weight_kg)


def calculate_size_maturation_clearance(
    weight_kg: float,
    pma_weeks: float,
    adult_clearance: float = DEFAULT_ADULT_CLEARANCE,
    pma50_weeks: float = DEFAULT_PMA50_WEEKS,
    hill: float = DEFAULT_HILL,
) -> float:
    """Calculate Model B: weight scaling multiplied by maturation."""
    weight_only_clearance = calculate_weight_only_clearance(
        weight_kg=weight_kg,
        adult_clearance=adult_clearance,
    )

    maturation_factor = calculate_maturation_factor(
        pma_weeks=pma_weeks,
        pma50_weeks=pma50_weeks,
        hill=hill,
    )

    return weight_only_clearance * maturation_factor