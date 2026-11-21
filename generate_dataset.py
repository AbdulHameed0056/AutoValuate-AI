"""Generate a realistic synthetic used-car dataset grounded in domain depreciation curves."""

from pathlib import Path
import csv
import random

from car_catalog import (
    CAR_CATALOG,
    CONDITION_OPTIONS,
    CURRENT_YEAR,
    OWNER_OPTIONS,
    SELLER_OPTIONS,
)

FUEL_PRICE_MULTIPLIERS = {
    "Petrol": 1.00,
    "CNG": 0.96,
    "Diesel": 1.08,
    "Hybrid": 1.15,
    "Electric": 1.12,
}

TRANSMISSION_MULTIPLIERS = {
    "Manual": 1.00,
    "Automatic": 1.11,
}

OWNER_MULTIPLIERS = {
    "First Owner": 1.00,
    "Second Owner": 0.89,
    "Third Owner": 0.78,
    "Fourth & Above": 0.67,
}

SELLER_MULTIPLIERS = {
    "Individual": 0.98,
    "Dealer": 1.00,
    "Trustmark Dealer": 1.05,
}

CONDITION_MULTIPLIERS = {
    "Excellent": 1.07,
    "Good": 1.00,
    "Fair": 0.88,
}


def compute_realistic_price(
    base_price: float,
    retention: float,
    age: int,
    km_driven: int,
    fuel_type: str,
    transmission: str,
    owner: str,
    seller_type: str,
    condition: str,
    trim_factor: float,
) -> int:
    """Estimate realistic used-car resale price in INR from vehicle attributes."""
    # Year-1 showroom depreciation is steeper (~14%), then follows retention curve.
    effective_retention = retention
    age_factor = (0.86 if age >= 1 else 0.96) * (effective_retention ** max(age - 1, 0))
    age_factor = max(age_factor, 0.14)

    # Kilometers driven penalty: steeper early wear, gentler slope above 120k km.
    expected_km = max(age, 1) * 11500
    km_ratio = km_driven / 100000.0
    excess_km_penalty = max(0.0, (km_driven - expected_km) / 100000.0) * 0.08
    km_factor = max(0.50, 1.0 - (0.19 * (km_ratio ** 0.82)) - excess_km_penalty)

    multiplier = (
        FUEL_PRICE_MULTIPLIERS[fuel_type]
        * TRANSMISSION_MULTIPLIERS[transmission]
        * OWNER_MULTIPLIERS[owner]
        * SELLER_MULTIPLIERS[seller_type]
        * CONDITION_MULTIPLIERS[condition]
        * trim_factor
    )

    raw_price = base_price * age_factor * km_factor * multiplier
    # Clamp minimum scrap/floor price to realistic value and round to nearest 1,000 INR.
    return max(85000, int(round(raw_price, -3)))


def generate_rows(samples_per_model: int = 85, seed: int = 42) -> list[dict[str, object]]:
    """Create balanced synthetic car listings across every brand and model."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for brand, models in CAR_CATALOG.items():
        for model_name, spec in models.items():
            for _ in range(samples_per_model):
                year = rng.randint(2011, CURRENT_YEAR - 1)
                age = CURRENT_YEAR - year

                # Correlate odometer reading realistically with vehicle age.
                annual_km = rng.randint(6000, 17500)
                km_driven = max(1800, min(240000, int(age * annual_km + rng.randint(-3000, 9000))))

                fuel_type = rng.choice(spec["fuels"])
                transmission = rng.choice(spec["transmissions"])

                # Newer cars are far more likely to be first-owner.
                if age <= 3:
                    owner = rng.choices(OWNER_OPTIONS, weights=[0.82, 0.15, 0.03, 0.0])[0]
                elif age <= 7:
                    owner = rng.choices(OWNER_OPTIONS, weights=[0.52, 0.34, 0.11, 0.03])[0]
                else:
                    owner = rng.choices(OWNER_OPTIONS, weights=[0.22, 0.40, 0.27, 0.11])[0]

                seller_type = rng.choices(SELLER_OPTIONS, weights=[0.48, 0.36, 0.16])[0]
                condition = rng.choices(CONDITION_OPTIONS, weights=[0.28, 0.54, 0.18])[0]

                # Trim level variation (+/- 6% on specs and base valuation).
                trim_factor = rng.uniform(0.94, 1.07)
                engine_cc = 0 if fuel_type == "Electric" else int(round(spec["engine_cc"] * rng.uniform(0.96, 1.04)))
                max_power_bhp = round(spec["max_power_bhp"] * trim_factor, 1)
                mileage_kmpl = round(
                    spec["mileage_kmpl"]
                    * (1.14 if fuel_type == "Diesel" else 1.25 if fuel_type == "Hybrid" else 1.0)
                    * rng.uniform(0.93, 1.07),
                    1,
                )

                market_noise = rng.gauss(1.0, 0.035)
                selling_price = compute_realistic_price(
                    base_price=spec["base_price"],
                    retention=spec["retention"],
                    age=age,
                    km_driven=km_driven,
                    fuel_type=fuel_type,
                    transmission=transmission,
                    owner=owner,
                    seller_type=seller_type,
                    condition=condition,
                    trim_factor=trim_factor * market_noise,
                )

                rows.append(
                    {
                        "brand": brand,
                        "model": model_name,
                        "body_type": spec["body_type"],
                        "year": year,
                        "km_driven": km_driven,
                        "fuel_type": fuel_type,
                        "transmission": transmission,
                        "owner": owner,
                        "seller_type": seller_type,
                        "condition": condition,
                        "engine_cc": engine_cc,
                        "max_power_bhp": max_power_bhp,
                        "mileage_kmpl": mileage_kmpl,
                        "selling_price": selling_price,
                    }
                )

    rng.shuffle(rows)
    return rows


def save_dataset(output_path: Path | None = None) -> Path:
    """Write the generated dataset to CSV and return its path."""
    target = output_path or Path(__file__).parent / "data" / "used_cars.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()

    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows at {target}")
    return target


if __name__ == "__main__":
    save_dataset()
