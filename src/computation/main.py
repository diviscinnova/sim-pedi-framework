# Run the SIM-PEDI illustrative analysis.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from computation.clearance import (
    calculate_size_maturation_clearance,
    calculate_weight_only_clearance,
)
from computation.maturation import (
    calculate_maturation_factor,
    calculate_pma_weeks,
)


REQUIRED_COLUMNS = {
    "profile_id",
    "postnatal_age_days",
    "gestational_age_at_birth_weeks",
    "weight_kg",
}

NUMERIC_COLUMNS = [
    "postnatal_age_days",
    "gestational_age_at_birth_weeks",
    "weight_kg",
]


def load_profiles(data_path: Path) -> pd.DataFrame:
    # Read and validate synthetic input profiles.
    df = pd.read_csv(
        data_path,
        dtype={"profile_id": "string"},
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The input dataset is empty.")

    profile_ids = df["profile_id"].str.strip()

    if profile_ids.isna().any() or profile_ids.eq("").any():
        raise ValueError("Profile IDs must be present and nonblank.")

    if profile_ids.duplicated().any():
        raise ValueError("Profile IDs must be unique.")

    df["profile_id"] = profile_ids

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} must contain finite, nonmissing numbers."
            )

    if (df["postnatal_age_days"] < 0).any():
        raise ValueError("Postnatal age cannot be negative.")

    if (df["gestational_age_at_birth_weeks"] <= 0).any():
        raise ValueError(
            "Gestational age at birth must be positive."
        )

    if (df["weight_kg"] <= 0).any():
        raise ValueError("Body weight must be positive.")

    return df


def run_analysis(profiles: pd.DataFrame) -> pd.DataFrame:
    # Calculate ages, maturation factors, and model comparisons.
    df = profiles.copy()

    df["pma_weeks"] = df.apply(
        lambda row: calculate_pma_weeks(
            postnatal_age_days=row["postnatal_age_days"],
            gestational_age_at_birth_weeks=(
                row["gestational_age_at_birth_weeks"]
            ),
        ),
        axis=1,
    )

    df["age_years"] = df["postnatal_age_days"] / 365.25

    df["maturation_factor"] = df["pma_weeks"].apply(
        calculate_maturation_factor
    )

    df["clearance_weight_only"] = df["weight_kg"].apply(
        calculate_weight_only_clearance
    )

    df["clearance_size_maturation"] = df.apply(
        lambda row: calculate_size_maturation_clearance(
            weight_kg=row["weight_kg"],
            pma_weeks=row["pma_weeks"],
        ),
        axis=1,
    )

    df["clearance_difference"] = (
        df["clearance_size_maturation"]
        - df["clearance_weight_only"]
    )

    df["percentage_difference"] = (
        100.0
        * df["clearance_difference"]
        / df["clearance_weight_only"]
    )

    output_columns = [
        "pma_weeks",
        "age_years",
        "maturation_factor",
        "clearance_weight_only",
        "clearance_size_maturation",
        "clearance_difference",
        "percentage_difference",
    ]

    if not np.isfinite(
        df[output_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Model outputs contain nonfinite values.")

    if not df["maturation_factor"].between(0, 1).all():
        raise ValueError("Maturation factors must be between 0 and 1.")

    clearance_columns = [
        "clearance_weight_only",
        "clearance_size_maturation",
    ]

    if (df[clearance_columns] <= 0).any().any():
        raise ValueError("Calculated clearance must be positive.")

    return df.sort_values(
        ["age_years", "profile_id"]
    ).reset_index(drop=True)


def save_figure(results: pd.DataFrame, figure_path: Path) -> None:
    """Show absolute predictions and their relative difference."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: absolute model predictions.
    axes[0].plot(
        results["age_years"],
        results["clearance_weight_only"],
        marker="o",
        linestyle="-",
        label="Model A: weight-only",
    )

    axes[0].plot(
        results["age_years"],
        results["clearance_size_maturation"],
        marker="s",
        linestyle="--",
        label="Model B: size + maturation",
    )

    axes[0].set_xlabel("Chronological age (years)")
    axes[0].set_ylabel("Simulated clearance (arbitrary units)")
    axes[0].set_title("Clearance predictions")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: positive reduction relative to Model A.
    reduction_percent = -results["percentage_difference"]

    axes[1].plot(
        results["age_years"],
        reduction_percent,
        marker="o",
        color="tab:green",
    )

    for age, reduction in zip(
        results["age_years"],
        reduction_percent,
    ):
        axes[1].annotate(
            f"{reduction:.3f}%",
            (age, reduction),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    axes[1].set_xlabel("Chronological age (years)")
    axes[1].set_ylabel("Reduction relative to Model A (%)")
    axes[1].set_title("Effect of the maturation multiplier")
    axes[1].set_ylim(
        bottom=0,
        top=max(1.0, float(reduction_percent.max()) * 1.2),
    )
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("SIM-PEDI: size scaling and maturation")

    fig.text(
        0.5,
        0.01,
        "Synthetic profiles; illustrative model; not for clinical dosing.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the illustrative SIM-PEDI comparison."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dummy_profiles.csv"),
        help="Input CSV path, relative to the current directory.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Output directory, relative to the current directory.",
    )

    args = parser.parse_args()

    profiles = load_profiles(args.data)
    results = run_analysis(profiles)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.results_dir / "model_comparison.csv"
    figure_path = args.results_dir / "clearance_comparison.png"

    results.to_csv(csv_path, index=False)
    save_figure(results, figure_path)

    print(f"Processed {len(results)} synthetic profiles.")
    print(f"Saved results to: {csv_path.resolve()}")
    print(f"Saved figure to: {figure_path.resolve()}")


if __name__ == "__main__":
    main()