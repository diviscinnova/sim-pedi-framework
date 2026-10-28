# Illustrative developmental maturation functions for SIM-PEDI.

from math import exp, isfinite, log


DEFAULT_PMA50_WEEKS = 47.7
DEFAULT_HILL = 3.40


def calculate_pma_weeks(
    postnatal_age_days: float,
    gestational_age_at_birth_weeks: float,
) -> float:
    # Calculate PMA from individual birth gestational age and age since birth.
    if (
        not isfinite(postnatal_age_days)
        or postnatal_age_days < 0
    ):
        raise ValueError(
            "Postnatal age must be finite and nonnegative."
        )

    if (
        not isfinite(gestational_age_at_birth_weeks)
        or gestational_age_at_birth_weeks <= 0
    ):
        raise ValueError(
            "Gestational age at birth must be finite and positive."
        )

    pma_weeks = (
        gestational_age_at_birth_weeks
        + postnatal_age_days / 7.0
    )

    if not isfinite(pma_weeks):
        raise ValueError("Calculated PMA must be finite.")

    return pma_weeks


def calculate_maturation_factor(
    pma_weeks: float,
    pma50_weeks: float = DEFAULT_PMA50_WEEKS,
    hill: float = DEFAULT_HILL,
) -> float:
    """
    Calculate an illustrative sigmoid maturation factor.

    F = PMA**hill / (PMA**hill + PMA50**hill)

    PMA and PMA50 are expressed in weeks. PMA50 is the PMA at
    which F equals 0.5. The defaults are demonstration parameters,
    not a claim of validity for every drug or population.
    """
    parameters = {
        "PMA": pma_weeks,
        "PMA50": pma50_weeks,
        "Hill coefficient": hill,
    }

    for name, value in parameters.items():
        if not isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be finite and positive."
            )

    # Equivalent log-domain form avoids large direct powers.
    z = hill * (log(pma_weeks) - log(pma50_weeks))

    if z >= 0:
        return 1.0 / (1.0 + exp(-z))

    exp_z = exp(z)
    return exp_z / (1.0 + exp_z)