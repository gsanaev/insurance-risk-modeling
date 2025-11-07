"""
simulate_data.py
----------------
Generates a synthetic auto insurance portfolio dataset following the
Poisson–Gamma frequency–severity model described in PROJECT_SETUP.md.

This version:
- Uses empirical reference distributions (MiD, KBA, GDV) from data/reference/
- Requires all reference CSVs to exist before execution
- Produces data/raw/synthetic_insurance_portfolio.csv

Execution:
    uv run python -m src.ins.simulate_data
"""

import numpy as np
import pandas as pd
import pathlib

def sample_age_from_bins(rng, age_bins: np.ndarray, age_probs: np.ndarray, size: int) -> np.ndarray:
    """
    Sample ages using MiD age bins with the last bin open to 80.
    age_bins: starts (e.g., [18,23,28,...,58])
    """
    uppers = np.append(age_bins[1:], 80)                   # last bin goes to 80
    idx = rng.choice(len(age_bins), size=size, p=age_probs)
    low = age_bins[idx]
    high = uppers[idx]
    width = np.maximum(high - low, 1)
    u = rng.random(size)
    return np.clip((low + np.floor(u * width)).astype(int), 18, 80)


def regional_vehicle_probs(base_probs: pd.Series, mix_row: pd.Series | None) -> np.ndarray:
    """
    Multiply national vehicle-type probabilities by regional multipliers (if given),
    then renormalize.
    """
    p = base_probs.to_numpy().astype(float)
    if mix_row is not None:
        adj = mix_row.to_numpy(dtype=float)
        p = p * adj
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


# ==============================================================
# STEP 0 — RANDOM SEED HANDLER
# ==============================================================
def set_seed(seed: int = 42) -> np.random.Generator:
    """Return a reproducible random generator."""
    return np.random.default_rng(seed)


# ==============================================================
# STEP 1 — LOAD REFERENCE DISTRIBUTIONS (MiD, KBA, GDV)
# ==============================================================
def load_reference_distributions(ref_dir: pathlib.Path) -> dict:
    """
    Load reference CSVs created in Notebook 1a.
    If any are missing, raise a FileNotFoundError.
    """
    required_files = [
        "vehicle_type_distribution.csv",
        "driver_age_distribution.csv",
        "annual_mileage_distribution.csv",
        "claim_stats_reference.csv",
        "region_pkw_distribution.csv",
        "region_frequency_adj.csv",
        "vehicle_frequency_adj.csv",
        "region_severity_adj.csv",
        "vehicle_severity_adj.csv",
    ]

    missing = [f for f in required_files if not (ref_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"❌ Missing reference files: {', '.join(missing)}\n"
            "Please run Notebook 1a — Reference Data Exploration (MiD, KBA, GDV) "
            "to generate these CSVs before running simulate_data.py."
        )

    print("✅ All reference datasets found.")
    ref = {
        "vehicle": pd.read_csv(ref_dir / "vehicle_type_distribution.csv"),
        "age": pd.read_csv(ref_dir / "driver_age_distribution.csv"),
        "mileage": pd.read_csv(ref_dir / "annual_mileage_distribution.csv"),
        "claims": pd.read_csv(ref_dir / "claim_stats_reference.csv"),
        "region": pd.read_csv(ref_dir / "region_pkw_distribution.csv"),
        "reg_freq": pd.read_csv(ref_dir / "region_frequency_adj.csv"),
        "vt_freq": pd.read_csv(ref_dir / "vehicle_frequency_adj.csv"),
        "reg_sev": pd.read_csv(ref_dir / "region_severity_adj.csv"),
        "vt_sev": pd.read_csv(ref_dir / "vehicle_severity_adj.csv"),
    }
    return ref


# ==============================================================
# STEP 2 — GENERATE BASE COVARIATES (region-first, stratified)
# ==============================================================
def generate_covariates(n: int, rng: np.random.Generator, ref: dict) -> pd.DataFrame:
    """
    1) Allocate sample across regions using KBA shares.
    2) Within each region, sample from national reference distributions (MiD/KBA).
    3) Concatenate strata to preserve national marginals and regional totals.
    """
    regions_ref = ref["region"].copy()
    region_codes = regions_ref["region"].to_numpy()
    region_probs = regions_ref["pkw_share"].to_numpy()
    region_probs = region_probs / region_probs.sum()

    # ---- 1) Allocate region counts (multinomial)
    counts = rng.multinomial(n, region_probs)
    diff = n - counts.sum()
    if diff != 0:
        counts[np.argmax(counts)] += diff

    # National priors
    age_bins = ref["age"]["age_bin"].to_numpy()
    age_probs = ref["age"]["probability"].to_numpy()

    vref = ref["vehicle"]
    base_vehicle_probs = vref.set_index("vehicle_type")["probability"]
    mean_age_map  = vref.set_index("vehicle_type")["mean_vehicle_age"].to_dict()
    mean_power_map = vref.set_index("vehicle_type")["mean_engine_power_kw"].to_dict()
    vehicle_types = base_vehicle_probs.index.to_list()

    mileage_means = ref["mileage"]["mean_km"].to_numpy()
    mileage_probs = ref["mileage"]["probability"].to_numpy()

    # Optional regional multipliers for vehicle mix (can be all 1.0)
    mix_ref = None
    if "region_mix" in ref:
        mix_ref = ref["region_mix"].set_index("region")

    frames = []
    for r, m in zip(region_codes, counts):
        if m == 0:
            continue

        # 2a) Driver age & experience (MiD)
        ages = sample_age_from_bins(rng, age_bins, age_probs, m)
        years_licensed = np.clip(ages - rng.integers(16, 20, m), 0, 60)

        # 2b) Vehicle type (national mix, optionally region multipliers)
        mix_row = mix_ref.loc[r] if (mix_ref is not None and r in mix_ref.index) else None
        p_region = regional_vehicle_probs(base_vehicle_probs, mix_row)
        vtypes = rng.choice(vehicle_types, size=m, p=p_region)

        # ---- UPDATED: smooth vehicle_age to avoid empty integer bins
        v_age = np.clip(
            np.round([
                rng.normal(mean_age_map[v], 2.0) + rng.uniform(-0.3, 0.3)
                for v in vtypes
            ]),
            0, None
        ).astype(int)

        # Engine power (unchanged, ensure >= 50 kW)
        v_power = np.clip(
            [rng.normal(mean_power_map[v], 20.0) for v in vtypes],
            50, None
        ).astype(int)

        # 2c) Mileage (MiD)
        mileage = rng.choice(mileage_means, size=m, p=mileage_probs)
        mileage = np.round(mileage * rng.normal(1.0, 0.15, m)).astype(int)

        # 2d) Fixed attributes
        urban_density = regions_ref.loc[regions_ref["region"] == r, "urban_density"].iloc[0]
        exposure = np.where(rng.random(m) < 0.85, 1.0, rng.choice([0.5, 0.75], size=m))
        garage = rng.random(m) < 0.55
        bonus_malus = np.clip(rng.normal(1.0, 0.15, m), 0.6, 1.6)
        prior = rng.poisson(0.15, m)
        prior[rng.random(m) < 0.6] = 0
        prior = np.clip(prior, 0, 3)
        commercial_use = rng.random(m) < 0.08
        telematics = rng.random(m) < 0.20
        sum_insured = np.clip(rng.normal(40000, 12000, m), 5000, 80000)
        policy_year = rng.choice([2023, 2024, 2025], size=m)

        frames.append(pd.DataFrame({
            "region": r,
            "urban_density": urban_density,
            "driver_age": ages,
            "years_licensed": years_licensed,
            "vehicle_type": vtypes,
            "vehicle_age": v_age,
            "engine_power_kw": v_power,
            "annual_mileage_km": mileage,
            "exposure": exposure,
            "garage": garage,
            "bonus_malus": bonus_malus,
            "prior_claims_3y": prior,
            "commercial_use": commercial_use,
            "telematics_opt_in": telematics,
            "sum_insured": sum_insured,
            "policy_year": policy_year,
        }))

    df = pd.concat(frames, ignore_index=True)

    # Assign policy_id after concatenation
    df["policy_id"] = [f"P{i:07d}" for i in range(1, len(df) + 1)]
    return df[[
        "policy_id","exposure","driver_age","years_licensed",
        "vehicle_age","vehicle_type","engine_power_kw",
        "annual_mileage_km","region","urban_density","garage",
        "bonus_malus","prior_claims_3y","commercial_use",
        "telematics_opt_in","sum_insured","policy_year"
    ]]


# ==============================================================
# STEP 3 — SIMULATE CLAIM FREQUENCY & SEVERITY
# ==============================================================
def simulate_claims(df: pd.DataFrame, alpha=2.0, rng=None, ref=None) -> pd.DataFrame:
    if rng is None:
        rng = set_seed(42)

    # --- Frequency linear predictor (before calibrations)
    log_lambda = (
        -2.70
        + 0.002 * (df["driver_age"] - 45)
        + 0.18 * (df["annual_mileage_km"] / 10_000)
        + 0.25 * (df["vehicle_type"].eq("sports"))
        + 0.05 * (df["vehicle_type"].eq("SUV"))
        - 0.05 * df["garage"].astype(int)
        + 0.6  * (df["bonus_malus"] - 1)
        + 0.2  * df["prior_claims_3y"]
        + 0.15 * df["commercial_use"].astype(int)
        - 0.10 * df["telematics_opt_in"].astype(int)
    )

    # --- Severity linear predictor (before calibrations)
    log_mu = (
        7.55
        + 0.015 * df["vehicle_age"]
        + 0.28  * df["vehicle_type"].eq("sports")
        + 0.10  * df["vehicle_type"].eq("SUV")
        + 0.06  * (df["engine_power_kw"] / 50)
        + 0.04  * (df["sum_insured"] / 10_000)
        - 0.03  * df["garage"].astype(int)
        + 0.08  * df["commercial_use"].astype(int)
    )

    # ---------- NEW: portfolio calibration to targets ----------
    # Fallback if ref not passed
    target_freq = 0.08
    target_sev  = 2700.0
    if ref is not None and "claims" in ref:
        target_freq = float(ref["claims"].iloc[0]["mean_claim_frequency"])
        target_sev  = float(ref["claims"].iloc[0]["mean_severity_eur"])

    # current λ & μ (pre-calibration)
    lam_pre = np.exp(log_lambda) * df["exposure"].to_numpy()
    mu_pre  = np.exp(log_mu)

    # expected portfolio frequency (per policy-year)
    freq_hat = lam_pre.sum() / df["exposure"].sum()
    # scale λ so mean frequency hits target
    k_lambda = target_freq / max(freq_hat, 1e-12)
    log_lambda = log_lambda + np.log(k_lambda)

    # expected portfolio mean severity (weighted by expected claims)
    sev_hat = (mu_pre * lam_pre).sum() / max(lam_pre.sum(), 1e-12)
    # scale μ so claim-weighted mean severity hits target
    k_mu = target_sev / max(sev_hat, 1e-12)
    log_mu = log_mu + np.log(k_mu)
    # -----------------------------------------------------------

    # sample counts with calibrated λ
    num_claims = rng.poisson(np.exp(log_lambda) * df["exposure"].to_numpy())

    # sample severities with calibrated μ
    mu = np.exp(log_mu)
    totals = np.zeros(len(df))
    avgs   = np.zeros(len(df))
    for i, n in enumerate(num_claims):
        if n > 0:
            sev = rng.gamma(shape=alpha, scale=mu[i] / alpha, size=int(n))
            totals[i] = sev.sum()
            avgs[i]   = sev.mean()
    df["num_claims"] = num_claims
    df["total_claim_amount"] = np.round(totals, 2)
    df["avg_claim_amount"]   = np.round(avgs, 2)
    return df


# ==============================================================
# STEP 4 — MAIN ENTRY POINT
# ==============================================================
def main():
    """Run full synthetic portfolio simulation using reference data."""
    n = 100_000
    seed = 42
    alpha = 2.0

    rng = set_seed(seed)
    ref_dir = pathlib.Path("data/reference")
    raw_dir = pathlib.Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ref = load_reference_distributions(ref_dir)
    print("📚 Reference datasets successfully loaded from:", ref_dir)

    df = generate_covariates(n=n, rng=rng, ref=ref)
    df = simulate_claims(df, ref=ref, alpha=alpha, rng=rng)

    out_path = raw_dir / "synthetic_insurance_portfolio.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df):,} simulated policies to {out_path}")
    print(f"🎲 Parameters: n={n}, seed={seed}, alpha={alpha}")


if __name__ == "__main__":
    main()
