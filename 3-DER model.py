"""
three-DER fairness-constrained Volt--Watt optimisation model.

1. Generate 100 correlated probabilistic operating scenarios.
2. Solve an unconstrained multi-scenario LinDistFlow Volt--Watt OPF benchmark.
3. Search the three pairwise fairness-stick coefficients with:
   - a coarse-to-fine Cartesian Grid Search, and
   - Gaussian-process Bayesian Optimisation with Expected Improvement.
4. Select the highest-production evaluated candidate that satisfies the target
   average-utilisation disparity whenever such a candidate is found.
5. Validate the fixed LinDistFlow DER operating points with an SOC branch-flow
   formulation and save detailed scenario/node/branch diagnostics and figures.

The three fairness-stick coefficients describe the pairwise relationships
between DER nodes 3, 6 and 12:

    V_shift_sq[3] <= k_36   * V_shift_sq[6]
    V_shift_sq[3] <= k_3_12 * V_shift_sq[12]
    V_shift_sq[6] <= k_6_12 * V_shift_sq[12]

The coefficients are outer-search parameters. During each inner OPF solve they
are fixed constants, while the DER shift parameters remain OPF decision
variables.
"""


import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import product

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scikit-optimize provides a maintained Gaussian-process Bayesian optimiser
# with an ask-and-tell interface suitable for expensive black-box OPF solves.
try:
    from skopt import Optimizer
    from skopt.space import Real
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "This program requires scikit-optimize. Install it with: "
        "python3 -m pip install scikit-optimize"
    ) from exc


# ============================================================
# Study settings
# ============================================================

OUTPUT_DIR = Path("results_three_DER_final_calibrated_correlation_grid_vs_bayesian_complete")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 13
SLACK = 0
DER_NODES = [3, 6, 12]
LOAD_NODES = [1, 3, 4, 5, 6, 8, 9, 11, 12]

# Probabilistic operating scenarios use exactly the same ranges as the
# previous deterministic 10 x 10 scenario grid. Every non-zero nodal load is
# sampled separately, while nodal demands remain strongly correlated. The
# slack-bus voltage is mildly negatively correlated with aggregate demand. A
# fixed seed makes all three optimisation cases use the same scenario set.
SLACK_VOLTAGE_MIN = 0.98
SLACK_VOLTAGE_MAX = 1.03
LOAD_FACTOR_MIN = 0.20
LOAD_FACTOR_MAX = 0.80

N_SCENARIOS = int(os.environ.get("VW_N_SCENARIOS", "100"))
SCENARIO_RANDOM_SEED = int(os.environ.get("VW_SCENARIO_RANDOM_SEED", "2026"))
SCENARIO_DISTRIBUTION = os.environ.get(
    "VW_SCENARIO_DISTRIBUTION", "correlated_truncated_normal"
).strip().lower()

# Scenario correlations requested for the final study.
# Each non-zero nodal load is sampled separately, with a strong positive
# equicorrelation between nodal load factors. The slack-bus voltage has a mild
# negative correlation with the aggregate load level, representing an LV
# feeder located electrically downstream of the regulated MV source: when the
# local demand is high, the feeder-head voltage tends to be lower.
TARGET_NODAL_LOAD_CORRELATION = float(os.environ.get(
    "VW_TARGET_NODAL_LOAD_CORRELATION", "0.75"
))
TARGET_SLACK_AGGREGATE_LOAD_CORRELATION = float(os.environ.get(
    "VW_TARGET_SLACK_LOAD_CORRELATION", "-0.25"
))
SLACK_CORRELATION_CALIBRATION_TOLERANCE = float(os.environ.get(
    "VW_SLACK_CORRELATION_TOLERANCE", "0.005"
))
SLACK_CORRELATION_CALIBRATION_GRID_SIZE = int(os.environ.get(
    "VW_SLACK_CORRELATION_GRID_SIZE", "81"
))

# Defaults place approximately 99.7% of an untruncated normal distribution
# inside the original range. Rejection sampling, rather than clipping, ensures
# every retained value lies inside the range without creating artificial mass
# at the endpoints.
SLACK_VOLTAGE_MEAN = float(os.environ.get(
    "VW_SLACK_VOLTAGE_MEAN",
    str((SLACK_VOLTAGE_MIN + SLACK_VOLTAGE_MAX) / 2.0),
))
SLACK_VOLTAGE_STD = float(os.environ.get(
    "VW_SLACK_VOLTAGE_STD",
    str((SLACK_VOLTAGE_MAX - SLACK_VOLTAGE_MIN) / 6.0),
))
LOAD_FACTOR_MEAN = float(os.environ.get(
    "VW_LOAD_FACTOR_MEAN",
    str((LOAD_FACTOR_MIN + LOAD_FACTOR_MAX) / 2.0),
))
LOAD_FACTOR_STD = float(os.environ.get(
    "VW_LOAD_FACTOR_STD",
    str((LOAD_FACTOR_MAX - LOAD_FACTOR_MIN) / 6.0),
))

PV_CAPACITY_PER_DER_MW = 10.0

# Maximum allowed difference between the average utilisations of any
# two participating DERs. The default value 0.05 means 5 percentage points.


# Fairness-stick coefficient bounds. Each k is an outer-search variable.
# Values closer to 1 impose a weaker proportional restriction between the
# corresponding DER shift parameters; smaller values impose a stronger one.
FAIRNESS_STICK_MAX = float(
    os.environ.get("VW_FAIRNESS_STICK_MAX", "1.0")
)
FAIRNESS_STICK_MIN = float(
    os.environ.get("VW_FAIRNESS_STICK_MIN", "0.97")
)
# Bayesian-optimisation controls. Each objective evaluation solves the complete
# joint OPF across all operating scenarios. The black-box fairness condition is
# represented by a large, continuous penalty during surrogate fitting, while
# the final reported solution is always selected using the explicit disparity
# condition rather than the penalised objective.
BO_INITIAL_EVALUATIONS = int(
    os.environ.get("VW_BO_INITIAL_EVALUATIONS", "12")
)
BO_ITERATIONS = int(
    os.environ.get("VW_BO_ITERATIONS", "28")
)
BO_RANDOM_SEED = int(
    os.environ.get("VW_BO_RANDOM_SEED", "42")
)
BO_ACQ_RESTARTS = int(
    os.environ.get("VW_BO_ACQ_RESTARTS", "8")
)
BO_ACQ_SAMPLES = int(
    os.environ.get("VW_BO_ACQ_SAMPLES", "10000")
)
BO_DUPLICATE_TOLERANCE = float(
    os.environ.get("VW_BO_DUPLICATE_TOLERANCE", "1e-7")
)
# Early stopping is evaluated only after the initial design and a minimum
# number of Bayesian steps. It stops when the best feasible production has not
# improved by at least BO_EARLY_STOP_TOLERANCE_MW for BO_EARLY_STOP_PATIENCE
# consecutive Bayesian evaluations. Set VW_BO_EARLY_STOP=0 to disable it.
BO_EARLY_STOP = os.environ.get("VW_BO_EARLY_STOP", "1") == "1"
BO_EARLY_STOP_PATIENCE = int(
    os.environ.get("VW_BO_EARLY_STOP_PATIENCE", "8")
)
BO_EARLY_STOP_MIN_BAYESIAN_EVALUATIONS = int(
    os.environ.get("VW_BO_EARLY_STOP_MIN_BAYESIAN_EVALUATIONS", "12")
)
BO_EARLY_STOP_TOLERANCE_MW = float(
    os.environ.get("VW_BO_EARLY_STOP_TOLERANCE_MW", "0.25")
)
# Infeasible observations receive this offset plus a violation-dependent term.
# The offset is deliberately larger than the full production range, ensuring
# that a feasible observation is preferred over an infeasible one.
BO_INFEASIBLE_OFFSET = float(
    os.environ.get("VW_BO_INFEASIBLE_OFFSET", "10000.0")
)
BO_VIOLATION_PENALTY = float(
    os.environ.get("VW_BO_VIOLATION_PENALTY", "100000.0")
)
# Final feasibility test used by both outer search methods. A value of 0.05
# means that the difference between the highest and lowest DER *average*
# utilisation must not exceed five percentage points. This is checked after
# each complete multi-scenario OPF solve.
FAIRNESS_STICK_TARGET_DISPARITY = float(
    os.environ.get("VW_TARGET_DISPARITY", "0.05")
)

# Grid-search benchmark controls. These reproduce the previous coarse-to-fine
# search and use exactly the same inner OPF as Bayesian optimisation.
RUN_GRID_SEARCH = os.environ.get("VW_RUN_GRID_SEARCH", "1") == "1"
GRID_COARSE_STEPS = int(os.environ.get("VW_GRID_COARSE_STEPS", "4"))
GRID_TOP_CENTRES = int(os.environ.get("VW_GRID_TOP_CENTRES", "1"))
GRID_FINE_STEP = float(os.environ.get("VW_GRID_FINE_STEP", "0.001"))
GRID_FINE_RADIUS = int(os.environ.get("VW_GRID_FINE_RADIUS", "1"))

V_MIN = 0.95
V_MAX = 1.05

# squared-voltage Volt-Watt formulation.
V1_SQ = 1.04 ** 2
V2_SQ = V_MAX ** 2
VSHIFT_MIN_SQ = 1.01 ** 2
VSHIFT_MAX_SQ = 1.05 ** 2

ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")

# Numerical tolerances used only for post-processing/diagnostics. They are not
# additional physical OPF constraints. In particular, SOC_TIGHTNESS_TOL is used
# to classify a scenario as numerically tight when its maximum absolute cone gap
# is no larger than 1e-5.
VOLTAGE_BINDING_TOL = 1e-4
CAPACITY_BINDING_TOL = 1e-5
VOLT_WATT_BINDING_TOL = 1e-5
SOC_TIGHTNESS_TOL = 1e-5

# Optional execution controls. The dissertation results use SOC validation and
# non-verbose solver output by default.
RUN_SOC_VALIDATION = os.environ.get("VW_RUN_SOC", "1") == "1"
SOLVER_VERBOSE = os.environ.get("VW_SOLVER_VERBOSE", "0") == "1"


# ============================================================
# Feeder data copied from the supplied baseline code
# ============================================================

base_l_P = 8.0 * np.array([
    0, 0.2, 0, 0.4, 0.17, 0.23, 1.155,
    0, 0.17, 0.843, 0, 0.17, 0.128
])

base_l_Q = 8.0 * np.array([
    0, 0.116, 0, 0.29, 0.125, 0.132,
    0.66, 0, 0.151, 0.462, 0, 0.08, 0.086
])

r = np.array([
[0, 0.007547918, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0041, 0, 0.007239685, 0, 0.007547918, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.0041, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004343811, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.003773959, 0.003773959, 0, 0.004322245, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00434686, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004343157, 0.01169764],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

x = np.array([
[0, 0.022173236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0064, 0, 0.007336076, 0, 0.022173236, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.0064, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004401645, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.011086618, 0.011086618, 0, 0.004433667, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002430473, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004402952, 0.004490848],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])


A = np.array([
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Preserve the supplied branch parameters for traceability, then construct the
# modified LV-feeder impedance set used throughout the study. Resistance and
# reactance are swapped and both are scaled by 0.075, producing a more
# resistance-dominant feeder while retaining the original branch pattern.
r_original = r.copy()
x_original = x.copy()
r = 0.075 * x_original.copy()
x = 0.075 * r_original.copy()

edges = [(i, j) for i in range(N) for j in range(N) if A[i, j] == 1]
edge_set = set(edges)

children = {i: [] for i in range(N)}
parent = {}
for i, j in edges:
    children[i].append(j)
    parent[j] = i

pv_capacity = np.zeros(N)
for node in DER_NODES:
    pv_capacity[node] = PV_CAPACITY_PER_DER_MW


# ============================================================
# Scenario construction
# ============================================================



def _sample_correlated_nodal_loads_and_slack(
    rng: np.random.Generator,
    size: int,
    nodal_load_correlation: float,
    slack_aggregate_correlation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample correlated nodal load factors and slack voltages robustly.

    A one-factor Gaussian model is used for the nodal loads. For node ``i``,

        z_i = sqrt(rho_load) * z_common
              + sqrt(1 - rho_load) * epsilon_i,

    so the latent pairwise correlation between any two nodal load factors is
    exactly ``rho_load`` before range truncation. The aggregate latent loading
    level is then standardised, and the slack-voltage latent variable is
    generated as

        z_slack = rho_slack * z_aggregate
                  + sqrt(1 - rho_slack**2) * epsilon_slack.

    This gives the requested negative correlation between slack voltage and
    aggregate loading without constructing a high-dimensional covariance
    matrix. Complete scenario vectors are accepted only when every nodal load
    factor and the slack voltage lie inside the original scenario-grid ranges.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0.0 <= nodal_load_correlation < 0.999:
        raise ValueError(
            "VW_TARGET_NODAL_LOAD_CORRELATION must lie in [0, 0.999)"
        )
    if not -0.999 < slack_aggregate_correlation < 0.999:
        raise ValueError(
            "VW_TARGET_SLACK_LOAD_CORRELATION must lie between -0.999 and 0.999"
        )
    if SLACK_VOLTAGE_STD <= 0.0 or LOAD_FACTOR_STD <= 0.0:
        raise ValueError("Scenario standard deviations must be positive")

    m = len(LOAD_NODES)
    sqrt_rho = float(np.sqrt(nodal_load_correlation))
    sqrt_one_minus_rho = float(np.sqrt(1.0 - nodal_load_correlation))
    sqrt_one_minus_slack_rho_sq = float(
        np.sqrt(1.0 - slack_aggregate_correlation**2)
    )

    accepted_loads: List[np.ndarray] = []
    accepted_slack: List[float] = []
    batch_size = max(2000, 20 * size)
    maximum_batches = 10000

    for _ in range(maximum_batches):
        if len(accepted_slack) >= size:
            break

        # Common system-wide loading tendency plus node-specific variation.
        z_common = rng.standard_normal(batch_size)
        z_idiosyncratic = rng.standard_normal((batch_size, m))
        z_load = (
            sqrt_rho * z_common[:, None]
            + sqrt_one_minus_rho * z_idiosyncratic
        )

        # Standardise the aggregate latent loading level analytically. For an
        # equicorrelated set of m unit-variance variables, the variance of the
        # mean is rho + (1-rho)/m.
        aggregate_latent = z_load.mean(axis=1)
        aggregate_std = np.sqrt(
            nodal_load_correlation
            + (1.0 - nodal_load_correlation) / m
        )
        z_aggregate = aggregate_latent / aggregate_std

        # Negative rho means high aggregate load tends to coincide with a lower
        # slack voltage, while independent noise preserves realistic scatter.
        z_slack = (
            slack_aggregate_correlation * z_aggregate
            + sqrt_one_minus_slack_rho_sq
            * rng.standard_normal(batch_size)
        )

        load_draws = LOAD_FACTOR_MEAN + LOAD_FACTOR_STD * z_load
        slack_draws = SLACK_VOLTAGE_MEAN + SLACK_VOLTAGE_STD * z_slack

        valid_loads = np.all(
            (load_draws >= LOAD_FACTOR_MIN)
            & (load_draws <= LOAD_FACTOR_MAX),
            axis=1,
        )
        valid = (
            valid_loads
            & (slack_draws >= SLACK_VOLTAGE_MIN)
            & (slack_draws <= SLACK_VOLTAGE_MAX)
            & np.isfinite(slack_draws)
            & np.all(np.isfinite(load_draws), axis=1)
        )

        if np.any(valid):
            valid_load_rows = load_draws[valid]
            valid_slack_rows = slack_draws[valid]
            remaining = size - len(accepted_slack)
            take = min(remaining, valid_slack_rows.size)
            accepted_loads.extend(
                np.asarray(valid_load_rows[:take], dtype=float)
            )
            accepted_slack.extend(
                np.asarray(valid_slack_rows[:take], dtype=float).tolist()
            )

    if len(accepted_slack) < size:
        raise RuntimeError(
            "Unable to generate enough valid correlated scenarios. "
            "Increase the load/slack standard deviations cautiously or widen "
            "the admissible scenario ranges."
        )

    return (
        np.asarray(accepted_slack[:size], dtype=float),
        np.vstack(accepted_loads[:size]).astype(float),
    )

def _calibrate_slack_load_correlation(
    target_correlation: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calibrate the latent slack-load coefficient to the realised sample.

    For each candidate latent coefficient, the same random seed is reused.
    Therefore, the underlying random draws are held fixed and only the latent
    correlation parameter changes. The candidate producing the realised
    Pearson correlation closest to the requested target is retained.
    """
    if not -0.95 <= target_correlation <= 0.95:
        raise ValueError("Target realised correlation must lie in [-0.95, 0.95]")
    if SLACK_CORRELATION_CALIBRATION_GRID_SIZE < 11:
        raise ValueError("VW_SLACK_CORRELATION_GRID_SIZE must be at least 11")

    # Search a broad latent range. The realised correlation is usually weaker
    # after truncation, so the latent value may need to be more negative than
    # the requested sample correlation.
    candidate_rhos = np.linspace(-0.95, 0.20, SLACK_CORRELATION_CALIBRATION_GRID_SIZE)
    best = None

    for latent_rho in candidate_rhos:
        rng = np.random.default_rng(SCENARIO_RANDOM_SEED)
        slack_values, nodal_load_factors = _sample_correlated_nodal_loads_and_slack(
            rng=rng,
            size=N_SCENARIOS,
            nodal_load_correlation=TARGET_NODAL_LOAD_CORRELATION,
            slack_aggregate_correlation=float(latent_rho),
        )
        aggregate_load = nodal_load_factors.mean(axis=1)
        realised = float(np.corrcoef(slack_values, aggregate_load)[0, 1])
        error = abs(realised - target_correlation)
        record = (error, slack_values, nodal_load_factors, float(latent_rho), realised)
        if best is None or error < best[0]:
            best = record

    assert best is not None

    # Refine locally around the best coarse latent coefficient.
    coarse_step = float(candidate_rhos[1] - candidate_rhos[0])
    centre = best[3]
    local_low = max(-0.999, centre - coarse_step)
    local_high = min(0.999, centre + coarse_step)
    local_rhos = np.linspace(local_low, local_high, 61)

    for latent_rho in local_rhos:
        rng = np.random.default_rng(SCENARIO_RANDOM_SEED)
        slack_values, nodal_load_factors = _sample_correlated_nodal_loads_and_slack(
            rng=rng,
            size=N_SCENARIOS,
            nodal_load_correlation=TARGET_NODAL_LOAD_CORRELATION,
            slack_aggregate_correlation=float(latent_rho),
        )
        aggregate_load = nodal_load_factors.mean(axis=1)
        realised = float(np.corrcoef(slack_values, aggregate_load)[0, 1])
        error = abs(realised - target_correlation)
        if error < best[0]:
            best = (error, slack_values, nodal_load_factors, float(latent_rho), realised)
        if error <= SLACK_CORRELATION_CALIBRATION_TOLERANCE:
            break

    _, slack_values, nodal_load_factors, latent_rho, realised = best
    return slack_values, nodal_load_factors, latent_rho, realised


def create_probabilistic_scenarios() -> pd.DataFrame:
    """Create a reproducible scenario set with calibrated correlation."""
    if N_SCENARIOS < 3:
        raise ValueError("N_SCENARIOS must be at least 3")
    if SCENARIO_DISTRIBUTION != "correlated_truncated_normal":
        raise ValueError(
            "The final model requires VW_SCENARIO_DISTRIBUTION="
            "'correlated_truncated_normal'"
        )

    (
        slack_values,
        nodal_load_factors,
        calibrated_latent_rho,
        calibrated_realised_rho,
    ) = _calibrate_slack_load_correlation(
        TARGET_SLACK_AGGREGATE_LOAD_CORRELATION
    )

    scenario_df = pd.DataFrame({
        "scenario": np.arange(N_SCENARIOS, dtype=int),
        "slack_voltage_pu": slack_values,
    })
    for column_index, node in enumerate(LOAD_NODES):
        scenario_df[f"load_factor_node_{node}"] = nodal_load_factors[:, column_index]

    factor_columns = [f"load_factor_node_{node}" for node in LOAD_NODES]
    scenario_df["load_factor"] = scenario_df[factor_columns].mean(axis=1)

    load_corr = scenario_df[factor_columns].corr(method="pearson")
    off_diagonal = load_corr.to_numpy()[np.triu_indices(len(LOAD_NODES), k=1)]
    realised_mean_pairwise = float(np.mean(off_diagonal))
    realised_min_pairwise = float(np.min(off_diagonal))
    realised_max_pairwise = float(np.max(off_diagonal))
    realised_slack_aggregate_pearson = float(
        scenario_df["slack_voltage_pu"].corr(
            scenario_df["load_factor"], method="pearson"
        )
    )
    realised_slack_aggregate_spearman = float(
        scenario_df["slack_voltage_pu"].corr(
            scenario_df["load_factor"], method="spearman"
        )
    )

    scenario_df["scenario_probability"] = 1.0 / float(N_SCENARIOS)
    scenario_df["sampling_distribution"] = SCENARIO_DISTRIBUTION
    scenario_df["sampling_seed"] = SCENARIO_RANDOM_SEED
    scenario_df["target_nodal_load_correlation"] = TARGET_NODAL_LOAD_CORRELATION
    scenario_df["target_slack_aggregate_load_correlation"] = (
        TARGET_SLACK_AGGREGATE_LOAD_CORRELATION
    )
    scenario_df["calibrated_latent_slack_load_correlation"] = calibrated_latent_rho
    scenario_df["calibration_absolute_error"] = abs(
        calibrated_realised_rho - TARGET_SLACK_AGGREGATE_LOAD_CORRELATION
    )
    scenario_df["realised_mean_pairwise_load_correlation"] = realised_mean_pairwise
    scenario_df["realised_min_pairwise_load_correlation"] = realised_min_pairwise
    scenario_df["realised_max_pairwise_load_correlation"] = realised_max_pairwise
    scenario_df["realised_pearson_slack_aggregate_load_correlation"] = (
        realised_slack_aggregate_pearson
    )
    scenario_df["realised_spearman_slack_aggregate_load_correlation"] = (
        realised_slack_aggregate_spearman
    )
    return scenario_df


def save_scenario_sampling_plots(
    scenario_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save diagnostics for aggregate and nodal scenario correlations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_columns = [f"load_factor_node_{node}" for node in LOAD_NODES]

    x = scenario_df["load_factor"].to_numpy(dtype=float)
    y = scenario_df["slack_voltage_pu"].to_numpy(dtype=float)
    realised_corr = float(np.corrcoef(x, y)[0, 1])
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.75)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    plt.plot(x_line, intercept + slope * x_line, linestyle="--", label="Linear trend")
    plt.xlim(LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
    plt.ylim(SLACK_VOLTAGE_MIN, SLACK_VOLTAGE_MAX)
    plt.xlabel("Mean nodal load factor")
    plt.ylabel("Slack voltage, p.u.")
    plt.title(f"Slack Voltage vs Aggregate Load (Pearson r = {realised_corr:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "slack_voltage_vs_aggregate_load_correlation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for node, column in zip(LOAD_NODES, factor_columns):
        plt.plot(scenario_df["scenario"], scenario_df[column], alpha=0.7, label=f"Node {node}")
    plt.xlabel("Scenario")
    plt.ylabel("Nodal load factor")
    plt.title("Separately Sampled but Correlated Nodal Load Factors")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "nodal_load_factors_by_scenario.png", dpi=300)
    plt.close()

    correlation_matrix = scenario_df[factor_columns].corr().to_numpy()
    plt.figure(figsize=(7, 6))
    image = plt.imshow(correlation_matrix, vmin=-1.0, vmax=1.0, aspect="equal")
    labels = [str(node) for node in LOAD_NODES]
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Load node")
    plt.ylabel("Load node")
    plt.title("Realised Nodal Load-Factor Correlations")
    plt.colorbar(image, label="Pearson correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "nodal_load_correlation_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(scenario_df["slack_voltage_pu"], bins=12, edgecolor="black")
    plt.axvline(SLACK_VOLTAGE_MIN, linestyle="--", label="Original range")
    plt.axvline(SLACK_VOLTAGE_MAX, linestyle="--")
    plt.xlabel("Slack voltage, p.u.")
    plt.ylabel("Number of scenarios")
    plt.title("Sampled Slack-Voltage Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sampled_slack_voltage_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(scenario_df["load_factor"], bins=12, edgecolor="black")
    plt.xlabel("Mean nodal load factor")
    plt.ylabel("Number of scenarios")
    plt.title("Sampled Aggregate Load-Factor Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "sampled_aggregate_load_factor_distribution.png", dpi=300)
    plt.close()

    scenario_df[factor_columns].corr().to_csv(
        output_dir / "realised_nodal_load_correlation_matrix.csv"
    )


def build_scenario_arrays(
    scenario_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario_count = len(scenario_df)

    load_p = np.zeros((scenario_count, N))
    load_q = np.zeros((scenario_count, N))
    pv_available = np.zeros((scenario_count, N))

    for s, row in scenario_df.iterrows():
        # Each non-zero load node has its own sampled scaling factor. Active and
        # reactive demand at that node use the same factor, preserving its base
        # power factor while allowing spatial demand diversity.
        for node in LOAD_NODES:
            factor = float(row[f"load_factor_node_{node}"])
            load_p[s, node] = base_l_P[node] * factor
            load_q[s, node] = base_l_Q[node] * factor
        pv_available[s, DER_NODES] = pv_capacity[DER_NODES]

    return load_p, load_q, pv_available


# ============================================================
# Generic solver helper
# ============================================================

def solve_cvxpy_problem(problem: cp.Problem) -> None:
    try:
        problem.solve(
            solver=cp.CLARABEL,
            verbose=SOLVER_VERBOSE,
            tol_gap_abs=1e-8,
            tol_gap_rel=1e-8,
            tol_feas=1e-8,
            max_iter=1000,
        )
    except Exception:
        problem.solve(
            solver=cp.SCS,
            verbose=SOLVER_VERBOSE,
            eps=1e-6,
            max_iters=300000,
        )


# ============================================================
# Joint LinDistFlow formulation
# ============================================================

def build_joint_local_model_problem(
    scenario_df: pd.DataFrame,
    formulation: str,
    fairness_stick_array: Optional[Sequence[float]] = None,
) -> Tuple[cp.Problem, Dict[str, object]]:
    """Build the joint multi-scenario LinDistFlow OPF used in the final study.

    Only two formulations are retained in this cleaned final model:

    ``utilisation``
        Unconstrained benchmark. The three DER-specific Volt--Watt shift
        parameters are free within their individual bounds and total DER
        production is maximised.

    ``fairness_stick``
        The same OPF objective and network equations are used, but the three
        pairwise fairness-stick inequalities are added. The coefficient values
        are supplied by the *outer* Grid Search or Bayesian Optimisation.

    A key modelling point is that the fairness-stick coefficients are fixed
    parameters during one OPF solve. Therefore, products such as
    ``k_36 * v_shift_sq[6]`` remain affine in the OPF decision variables and
    the inner optimisation remains continuous and convex.
    """
    valid_formulations = {"utilisation", "fairness_stick"}
    if formulation not in valid_formulations:
        raise ValueError(
            "formulation must be either 'utilisation' or 'fairness_stick'"
        )

    scenario_count = len(scenario_df)
    load_p, load_q, pv_available = build_scenario_arrays(scenario_df)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # One squared-voltage Volt--Watt shift parameter is shared across all
    # scenarios for each DER. This represents a fixed inverter setting that is
    # selected once and then applied under every sampled operating condition.
    v_shift_sq = cp.Variable(N, name="shared_v_shift_squared")

    # Network operating variables vary by scenario.
    p_der = cp.Variable(
        (scenario_count, N),
        nonneg=True,
        name="scenario_DER_active_power",
    )
    v_sq = cp.Variable(
        (scenario_count, N),
        name="scenario_voltage_squared",
    )

    # CVXPY support for native 3-D variables differs across versions, so branch
    # active/reactive flows are stored as one N x N matrix per scenario.
    P = [
        cp.Variable((N, N), name=f"P_s{s}")
        for s in range(scenario_count)
    ]
    Q = [
        cp.Variable((N, N), name=f"Q_s{s}")
        for s in range(scenario_count)
    ]

    constraints = []

    # Only DER buses require Volt--Watt shift variables. Non-DER entries are
    # fixed to zero so the N-element vector can be used safely throughout the
    # post-processing code without introducing unused degrees of freedom.
    for node in range(N):
        if node in DER_NODES:
            constraints.extend([
                v_shift_sq[node] >= VSHIFT_MIN_SQ,
                v_shift_sq[node] <= VSHIFT_MAX_SQ,
            ])
        else:
            constraints.append(v_shift_sq[node] == 0.0)

    # ------------------------------------------------------------------
    # Fairness-stick coupling (only for the fairness-constrained case)
    # ------------------------------------------------------------------
    if formulation == "fairness_stick":
        if fairness_stick_array is None:
            raise ValueError(
                "fairness_stick_array is required for fairness_stick"
            )

        stick_array = np.asarray(
            fairness_stick_array, dtype=float
        ).reshape(-1)
        if stick_array.size != 3:
            raise ValueError(
                "fairness_stick_array must contain exactly three values: "
                "[k_36, k_3_12, k_6_12]"
            )
        if np.any(stick_array <= 0.0) or np.any(stick_array > 1.0):
            raise ValueError(
                "Every fairness-stick coefficient must lie in (0, 1]"
            )

        k_36, k_3_12, k_6_12 = stick_array

        # Pairwise relationships between the three DER-specific shift settings.
        # Smaller k values restrict the upstream/earlier shift more strongly.
        constraints.extend([
            v_shift_sq[3] <= k_36 * v_shift_sq[6],
            v_shift_sq[3] <= k_3_12 * v_shift_sq[12],
            v_shift_sq[6] <= k_6_12 * v_shift_sq[12],
        ])

    # ------------------------------------------------------------------
    # Scenario-dependent feeder constraints
    # ------------------------------------------------------------------
    for s, row in scenario_df.iterrows():
        slack_voltage = float(row["slack_voltage_pu"])

        # The sampled feeder-head voltage is fixed for each scenario. All bus
        # voltages are constrained to the study limits.
        constraints.extend([
            v_sq[s, SLACK] == slack_voltage ** 2,
            v_sq[s, :] >= V_MIN ** 2,
            v_sq[s, :] <= V_MAX ** 2,
        ])

        for node in range(N):
            if node in DER_NODES:
                available = float(pv_available[s, node])

                # Physical PV availability limit.
                constraints.append(p_der[s, node] <= available)

                # Linear squared-voltage Volt--Watt upper bound. The separate
                # availability constraint above supplies the flat full-output
                # region when the voltage-dependent expression is larger than
                # the available PV power.
                constraints.append(
                    p_der[s, node]
                    <= available
                    * (v_shift_sq[node] - v_sq[s, node])
                    / (V2_SQ - V1_SQ)
                )
            else:
                constraints.append(p_der[s, node] == 0.0)

        # Entries of P and Q that do not correspond to physical feeder branches
        # are fixed to zero. This keeps the matrix representation unambiguous.
        for i in range(N):
            for j in range(N):
                if (i, j) not in edge_set:
                    constraints.extend([
                        P[s][i, j] == 0.0,
                        Q[s][i, j] == 0.0,
                    ])

        # LinDistFlow equations. DERs provide active power only in the final
        # study; reactive DER control and branch thermal constraints are not
        # included.
        for node in range(1, N):
            upstream = parent[node]

            if children[node]:
                downstream_p = cp.sum(cp.hstack([
                    P[s][node, child] for child in children[node]
                ]))
                downstream_q = cp.sum(cp.hstack([
                    Q[s][node, child] for child in children[node]
                ]))
            else:
                downstream_p = 0.0
                downstream_q = 0.0

            constraints.extend([
                # Active-power balance at the receiving bus.
                P[s][upstream, node]
                == load_p[s, node] - p_der[s, node] + downstream_p,

                # Reactive-power balance. There is no q_DER term.
                Q[s][upstream, node]
                == load_q[s, node] + downstream_q,

                # Linearised squared-voltage drop equation.
                v_sq[s, node]
                == v_sq[s, upstream]
                - 2.0 * (
                    r[upstream, node] * P[s][upstream, node]
                    + x[upstream, node] * Q[s][upstream, node]
                ),
            ])

    # The inner OPF objective is identical for the unconstrained and fairness-
    # stick cases. Fairness changes only the feasible set; it is not included as
    # a weighted term in the OPF objective.
    total_der_production = cp.sum(p_der[:, DER_NODES])
    objective = cp.Maximize(total_der_production)
    problem = cp.Problem(objective, constraints)

    variables = {
        "P": P,
        "Q": Q,
        "p_der": p_der,
        "v_sq": v_sq,
        "v_shift_sq": v_shift_sq,
        "load_p": load_p,
        "load_q": load_q,
        "pv_available": pv_available,
        "total_der_production": total_der_production,
    }
    return problem, variables




def extract_joint_solution(
    problem: cp.Problem,
    variables: Dict[str, object],
    scenario_count: int,
) -> Dict[str, object]:
    """Convert CVXPY variables into NumPy arrays and study-level metrics.

    The fairness metric used by the *outer* search is computed here from the
    solved DER outputs: for each DER, average utilisation is calculated over all
    scenarios, and disparity is the difference between the largest and smallest
    of those DER-average utilisations.
    """
    if problem.status not in ACCEPTED_STATUSES:
        return {
            "status": problem.status,
            "objective_value": np.nan,
        }

    P_value = np.stack([
        np.asarray(variable.value, dtype=float)
        for variable in variables["P"]
    ])
    Q_value = np.stack([
        np.asarray(variable.value, dtype=float)
        for variable in variables["Q"]
    ])
    v_sq_value = np.asarray(variables["v_sq"].value, dtype=float)
    p_der_value = np.asarray(variables["p_der"].value, dtype=float)
    shift_value = np.asarray(variables["v_shift_sq"].value, dtype=float)

    average_utilisation_by_node = {
        node: float(
            np.mean(
                p_der_value[:, node]
                / np.maximum(variables["pv_available"][:, node], 1e-12)
            )
        )
        for node in DER_NODES
    }
    achieved_average_disparity = float(
        max(average_utilisation_by_node.values())
        - min(average_utilisation_by_node.values())
    )

    return {
        "status": problem.status,
        "objective_value": float(problem.value),
        "P": P_value,
        "Q": Q_value,
        "p_der": p_der_value,
        "v_squared": v_sq_value,
        "voltage": np.sqrt(np.maximum(v_sq_value, 0.0)),
        "v_shift_squared": shift_value,
        "v_shift_pu": np.sqrt(np.maximum(shift_value, 0.0)),
        "load_p": variables["load_p"],
        "load_q": variables["load_q"],
        "pv_available": variables["pv_available"],
        "total_der_production_MW_scenarios": float(
            np.sum(p_der_value[:, DER_NODES])
        ),
        "average_utilisation_by_node": average_utilisation_by_node,
        "achieved_average_utilisation_disparity": achieved_average_disparity,
    }




def solve_utilisation_oriented_model(
    scenario_df: pd.DataFrame,
) -> Dict[str, object]:
    problem, variables = build_joint_local_model_problem(
        scenario_df=scenario_df,
        formulation="utilisation",
    )
    solve_cvxpy_problem(problem)

    return extract_joint_solution(
        problem,
        variables,
        len(scenario_df),
    )






def solve_single_fairness_stick(
    scenario_df: pd.DataFrame,
    fairness_stick_array: Sequence[float],
) -> Dict[str, object]:
    """
    Solve one pairwise fairness-stick problem.

    fairness_stick_array = [k_36, k_3_12, k_6_12].
    """
    stick_array = np.asarray(
        fairness_stick_array,
        dtype=float,
    ).reshape(3)

    problem, variables = build_joint_local_model_problem(
        scenario_df=scenario_df,
        formulation="fairness_stick",
        fairness_stick_array=stick_array,
    )
    solve_cvxpy_problem(problem)

    result = extract_joint_solution(
        problem,
        variables,
        len(scenario_df),
    )
    result["fairness_stick_array"] = stick_array
    result["k_36"] = float(stick_array[0])
    result["k_3_12"] = float(stick_array[1])
    result["k_6_12"] = float(stick_array[2])
    return result


def solve_fairness_stick_model(
    scenario_df: pd.DataFrame,
    target_disparity: float = FAIRNESS_STICK_TARGET_DISPARITY,
    n_initial: int = BO_INITIAL_EVALUATIONS,
    n_iterations: int = BO_ITERATIONS,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Run Gaussian-process Bayesian optimisation using scikit-optimize.

    The expensive black-box input is ``[k_36, k_3_12, k_6_12]``. Every
    evaluation solves the complete multi-scenario Volt-Watt OPF and returns
    total DER production and average utilisation disparity.

    ``skopt.Optimizer`` minimises a scalar merit function. Feasible points use
    negative production, so minimising the merit maximises production. An
    infeasible point receives a large offset and an additional term proportional
    to its disparity violation. This is the penalty-wrapper formulation proposed
    for the study. Importantly, the final solution is not selected from the
    penalised merit: it is the highest-production evaluated point satisfying the
    explicit condition ``disparity <= target_disparity``.
    """
    if target_disparity < 0.0:
        raise ValueError("target_disparity must be non-negative")
    if not 0.0 < FAIRNESS_STICK_MIN <= FAIRNESS_STICK_MAX <= 1.0:
        raise ValueError(
            "Require 0 < FAIRNESS_STICK_MIN <= FAIRNESS_STICK_MAX <= 1"
        )
    if n_initial < 5:
        raise ValueError("n_initial must be at least 5")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    lower = np.full(3, FAIRNESS_STICK_MIN, dtype=float)
    upper = np.full(3, FAIRNESS_STICK_MAX, dtype=float)

    dimensions = [
        Real(FAIRNESS_STICK_MIN, FAIRNESS_STICK_MAX,
             prior="uniform", transform="normalize", name="k_36"),
        Real(FAIRNESS_STICK_MIN, FAIRNESS_STICK_MAX,
             prior="uniform", transform="normalize", name="k_3_12"),
        Real(FAIRNESS_STICK_MIN, FAIRNESS_STICK_MAX,
             prior="uniform", transform="normalize", name="k_6_12"),
    ]
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        n_initial_points=n_initial,
        initial_point_generator="lhs",
        acq_func="EI",
        acq_optimizer="lbfgs",
        random_state=BO_RANDOM_SEED,
        acq_func_kwargs={"xi": 0.01},
        acq_optimizer_kwargs={
            "n_points": BO_ACQ_SAMPLES,
            "n_restarts_optimizer": BO_ACQ_RESTARTS,
            "n_jobs": 1,
        },
        model_queue_size=3,
    )

    rows: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    observed_x: List[np.ndarray] = []
    start_time = time.perf_counter()
    best_feasible_production = -np.inf
    no_improvement_count = 0
    stopped_early = False
    stopping_reason = "maximum evaluation budget reached"

    def is_duplicate(point: np.ndarray) -> bool:
        if not observed_x:
            return False
        distances = np.linalg.norm(
            np.asarray(observed_x) - point.reshape(1, -1), axis=1
        )
        return bool(np.min(distances) <= BO_DUPLICATE_TOLERANCE)

    # Penalised BO merit:
    # feasible:    merit = -production
    # infeasible:  merit = offset - production
    #                        + penalty * (disparity - target)
    #
    # Since skopt minimises the merit, lower values correspond to better candidates.
    def merit_value(production: float, disparity: float) -> float:
        violation = max(0.0, disparity - target_disparity)
        if violation <= 0.0:
            return -production
        return (
            BO_INFEASIBLE_OFFSET
            - production
            + BO_VIOLATION_PENALTY * violation
        )

    def evaluate(
        stick_array: Sequence[float],
        search_stage: str,
        acquisition_name: str,
    ) -> Tuple[bool, float]:
        nonlocal best_feasible_production, no_improvement_count
        k = np.clip(np.asarray(stick_array, dtype=float).reshape(3), lower, upper)
        evaluation_start = time.perf_counter()
        result = solve_single_fairness_stick(
            scenario_df=scenario_df,
            fairness_stick_array=k,
        )
        evaluation_seconds = time.perf_counter() - evaluation_start

        accepted = result["status"] in ACCEPTED_STATUSES
        if accepted:
            production = float(result["total_der_production_MW_scenarios"])
            disparity = float(result["achieved_average_utilisation_disparity"])
            accepted = np.isfinite(production) and np.isfinite(disparity)

        if accepted:
            feasible = disparity <= target_disparity + 1e-8
            merit = merit_value(production, disparity)
            v3 = float(result["v_shift_pu"][3])
            v6 = float(result["v_shift_pu"][6])
            v12 = float(result["v_shift_pu"][12])
        else:
            production = np.nan
            disparity = np.nan
            feasible = False
            # A failed OPF is worse than every normally infeasible observation.
            merit = BO_INFEASIBLE_OFFSET * 10.0
            v3 = v6 = v12 = np.nan

        candidate = len(rows)
        improvement = 0.0
        if feasible and np.isfinite(production):
            if np.isfinite(best_feasible_production):
                improvement = production - best_feasible_production
            else:
                improvement = np.inf
            if (not np.isfinite(best_feasible_production) or
                    production > best_feasible_production +
                    BO_EARLY_STOP_TOLERANCE_MW):
                best_feasible_production = production
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        elif search_stage == "bayesian":
            no_improvement_count += 1

        rows.append({
            "candidate": candidate,
            "search_stage": search_stage,
            "acquisition_function": acquisition_name,
            "k_36": float(k[0]),
            "k_3_12": float(k[1]),
            "k_6_12": float(k[2]),
            "fairness_stick_array": (
                f"[{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]"
            ),
            "status": result["status"],
            "V_shift_node_3_pu": v3,
            "V_shift_node_6_pu": v6,
            "V_shift_node_12_pu": v12,
            "total_DER_production_MW_scenarios": production,
            "average_utilisation_disparity": disparity,
            "target_disparity": float(target_disparity),
            "meets_target_disparity": feasible,
            "penalised_BO_objective": merit,
            "evaluation_time_seconds": evaluation_seconds,
            "cumulative_time_seconds": time.perf_counter() - start_time,
            "best_feasible_production_so_far_MW_scenarios": (
                best_feasible_production
                if np.isfinite(best_feasible_production) else np.nan
            ),
            "feasible_production_improvement_MW_scenarios": improvement,
            "early_stop_no_improvement_count": no_improvement_count,
        })
        results.append(result)
        observed_x.append(k.copy())

        production_text = f"{production:.4f}" if np.isfinite(production) else "nan"
        disparity_text = f"{disparity:.5f}" if np.isfinite(disparity) else "nan"
        print(
            f"Evaluation {candidate + 1:03d}: "
            f"k=[{k[0]:.5f}, {k[1]:.5f}, {k[2]:.5f}], "
            f"production={production_text}, disparity={disparity_text}, "
            f"feasible={feasible}, merit={merit:.4f}, "
            f"time={evaluation_seconds:.2f}s"
        )
        return accepted, float(merit)

    # Physically meaningful anchors make the comparison reproducible and ensure
    # that the known feasible neighbourhood is represented in the initial data.
    anchors = [
        [1.000, 1.000, 1.000],
        [0.990, 0.990, 0.990],
        [0.980, 0.980, 0.980],
        [0.970, 0.970, 0.970],
        [0.999, 0.989, 0.991],
    ]
    initial_count = min(len(anchors), n_initial)
    for point in anchors[:initial_count]:
        _, merit = evaluate(point, "initial", "anchor")
        optimizer.tell(point, merit)

    # Ask skopt for the remaining initial-design points. During this phase the
    # configured Latin-hypercube generator is used; afterwards ask() uses the GP.
    while len(rows) < n_initial:
        point = np.asarray(optimizer.ask(), dtype=float)
        retry = 0
        while is_duplicate(point) and retry < 100:
            point = np.asarray(optimizer.ask(), dtype=float)
            retry += 1
        _, merit = evaluate(point, "initial", "latin_hypercube")
        optimizer.tell(point.tolist(), merit)

    for bayesian_iteration in range(1, n_iterations + 1):
        point = np.asarray(optimizer.ask(), dtype=float)
        retry = 0
        while is_duplicate(point) and retry < 20:
            # Tell a duplicate a tiny deterministic repulsion value only to the
            # temporary copied optimiser would be cumbersome; instead jitter it
            # inside the very small coefficient domain.
            scale = (retry + 1) * 1e-5
            rng = np.random.default_rng(BO_RANDOM_SEED + len(rows) + retry)
            point = np.clip(point + rng.normal(0.0, scale, 3), lower, upper)
            retry += 1
        if is_duplicate(point):
            rng = np.random.default_rng(BO_RANDOM_SEED + len(rows))
            for _attempt in range(1000):
                candidate = rng.uniform(lower, upper)
                if not is_duplicate(candidate):
                    point = candidate
                    break

        _, merit = evaluate(point, "bayesian", "ExpectedImprovement")
        optimizer.tell(point.tolist(), merit)

        if (BO_EARLY_STOP
                and bayesian_iteration >= BO_EARLY_STOP_MIN_BAYESIAN_EVALUATIONS
                and np.isfinite(best_feasible_production)
                and no_improvement_count >= BO_EARLY_STOP_PATIENCE):
            stopped_early = True
            stopping_reason = (
                f"no feasible-production improvement greater than "
                f"{BO_EARLY_STOP_TOLERANCE_MW:.4f} MW for "
                f"{no_improvement_count} consecutive Bayesian evaluations"
            )
            print(
                f"Early stopping after {len(rows)} total evaluations: "
                f"{stopping_reason}."
            )
            break

    search_df = pd.DataFrame(rows)
    valid_indices = [
        i for i, row in enumerate(rows)
        if row["status"] in ACCEPTED_STATUSES
        and np.isfinite(row["total_DER_production_MW_scenarios"])
        and np.isfinite(row["average_utilisation_disparity"])
    ]
    if not valid_indices:
        return {
            "status": "all_bayesian_candidates_failed",
            "objective_value": np.nan,
        }, search_df

    feasible_indices = [
        i for i in valid_indices if bool(rows[i]["meets_target_disparity"])
    ]
    if feasible_indices:
        selected_index = max(
            feasible_indices,
            key=lambda i: (
                float(rows[i]["total_DER_production_MW_scenarios"]),
                -float(rows[i]["average_utilisation_disparity"]),
            ),
        )
        selection_rule = (
            "highest-production evaluated candidate satisfying the explicit "
            "average-utilisation-disparity constraint"
        )
    else:
        selected_index = min(
            valid_indices,
            key=lambda i: (
                float(rows[i]["average_utilisation_disparity"]),
                -float(rows[i]["total_DER_production_MW_scenarios"]),
            ),
        )
        selection_rule = (
            "minimum-disparity evaluated candidate because no point met the "
            "specified disparity limit"
        )

    selected_result = results[selected_index]
    selected_result["fairness_stick_target_disparity"] = float(target_disparity)
    selected_result["fairness_stick_selection_rule"] = selection_rule
    selected_result["fairness_stick_selected_index"] = int(selected_index)
    selected_result["fairness_stick_search_method"] = (
        "scikit-optimize Gaussian-process Bayesian optimisation with Expected "
        "Improvement and an outcome-constraint penalty wrapper; final selection "
        "uses the explicit utilisation-disparity constraint"
    )
    selected_result["bayesian_total_evaluations"] = len(rows)
    selected_result["bayesian_initial_evaluations"] = n_initial
    selected_result["bayesian_completed_sequential_evaluations"] = (
        len(rows) - n_initial
    )
    selected_result["bayesian_maximum_sequential_evaluations"] = n_iterations
    selected_result["bayesian_stopped_early"] = bool(stopped_early)
    selected_result["bayesian_stopping_reason"] = stopping_reason
    selected_result["bayesian_search_time_seconds"] = (
        time.perf_counter() - start_time
    )

    search_df["selected"] = False
    search_df.loc[selected_index, "selected"] = True
    return selected_result, search_df



# ============================================================
# Coarse-to-fine grid-search benchmark
# ============================================================

def solve_grid_search_fairness_stick_model(
    scenario_df: pd.DataFrame,
    target_disparity: float = FAIRNESS_STICK_TARGET_DISPARITY,
    number_of_steps: int = GRID_COARSE_STEPS,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Run the previous coarse-to-fine search over the three coefficients.

    Both this routine and the Bayesian optimiser call
    ``solve_single_fairness_stick``. The comparison therefore changes only the
    outer search strategy, while keeping the network, scenarios, OPF,
    disparity definition, coefficient bounds and solver settings identical.

    The coarse stage covers the complete three-dimensional coefficient range.
    The fine stage is centred on the most promising coarse region. A coarse
    point does not have to be feasible: when the coarse grid is too sparse to
    hit the narrow feasible region, the lowest-disparity coarse point is used
    as the refinement centre. This is the behaviour of the previous grid code.
    """
    if target_disparity < 0.0:
        raise ValueError("target_disparity must be non-negative")
    if number_of_steps < 2:
        raise ValueError("number_of_steps must be at least 2")
    if GRID_TOP_CENTRES < 1:
        raise ValueError("GRID_TOP_CENTRES must be at least 1")
    if GRID_FINE_STEP <= 0.0:
        raise ValueError("GRID_FINE_STEP must be positive")
    if GRID_FINE_RADIUS < 0:
        raise ValueError("GRID_FINE_RADIUS must be non-negative")

    coarse_values = np.linspace(
        FAIRNESS_STICK_MAX,
        FAIRNESS_STICK_MIN,
        number_of_steps,
    )
    rows: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    explored = set()
    search_start = time.perf_counter()
    best_feasible_production = -np.inf

    def evaluate(stick_array: Sequence[float], search_stage: str) -> None:
        nonlocal best_feasible_production
        k = np.asarray(stick_array, dtype=float).reshape(3)
        rounded = tuple(round(float(value), 7) for value in k)
        if rounded in explored:
            return
        explored.add(rounded)

        evaluation_start = time.perf_counter()
        result = solve_single_fairness_stick(
            scenario_df=scenario_df,
            fairness_stick_array=k,
        )
        evaluation_seconds = time.perf_counter() - evaluation_start

        accepted = result["status"] in ACCEPTED_STATUSES
        if accepted:
            production = float(result["total_der_production_MW_scenarios"])
            disparity = float(result["achieved_average_utilisation_disparity"])
            accepted = np.isfinite(production) and np.isfinite(disparity)

        if accepted:
            feasible = disparity <= target_disparity + 1e-8
            if feasible:
                best_feasible_production = max(
                    best_feasible_production,
                    production,
                )
            v3 = float(result["v_shift_pu"][3])
            v6 = float(result["v_shift_pu"][6])
            v12 = float(result["v_shift_pu"][12])
        else:
            production = np.nan
            disparity = np.nan
            feasible = False
            v3 = v6 = v12 = np.nan

        candidate = len(rows)
        rows.append({
            "candidate": candidate,
            "search_stage": search_stage,
            "k_36": float(k[0]),
            "k_3_12": float(k[1]),
            "k_6_12": float(k[2]),
            "fairness_stick_array": (
                f"[{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]"
            ),
            "status": result["status"],
            "V_shift_node_3_pu": v3,
            "V_shift_node_6_pu": v6,
            "V_shift_node_12_pu": v12,
            "total_DER_production_MW_scenarios": production,
            "average_utilisation_disparity": disparity,
            "target_disparity": float(target_disparity),
            "meets_target_disparity": feasible,
            "evaluation_time_seconds": evaluation_seconds,
            "cumulative_time_seconds": time.perf_counter() - search_start,
            "best_feasible_production_so_far_MW_scenarios": (
                best_feasible_production
                if np.isfinite(best_feasible_production) else np.nan
            ),
        })
        results.append(result)

        production_text = f"{production:.4f}" if np.isfinite(production) else "nan"
        disparity_text = f"{disparity:.5f}" if np.isfinite(disparity) else "nan"
        print(
            f"Grid evaluation {candidate + 1:03d}: "
            f"stage={search_stage}, "
            f"k=[{k[0]:.5f}, {k[1]:.5f}, {k[2]:.5f}], "
            f"production={production_text}, disparity={disparity_text}, "
            f"feasible={feasible}, time={evaluation_seconds:.2f}s"
        )

    def ranking_key(index: int) -> Tuple[float, float]:
        row = rows[index]
        return (
            float(row["total_DER_production_MW_scenarios"]),
            -float(row["average_utilisation_disparity"]),
        )

    # Stage 1: full Cartesian coarse grid.
    for stick_array in product(coarse_values, repeat=3):
        evaluate(stick_array, "coarse")

    valid_coarse = [
        i for i, row in enumerate(rows)
        if row["search_stage"] == "coarse"
        and row["status"] in ACCEPTED_STATUSES
        and np.isfinite(row["total_DER_production_MW_scenarios"])
        and np.isfinite(row["average_utilisation_disparity"])
    ]

    if valid_coarse:
        feasible_coarse = sorted(
            [i for i in valid_coarse if bool(rows[i]["meets_target_disparity"])],
            key=ranking_key,
            reverse=True,
        )
        lowest_disparity = sorted(
            valid_coarse,
            key=lambda i: (
                float(rows[i]["average_utilisation_disparity"]),
                -float(rows[i]["total_DER_production_MW_scenarios"]),
            ),
        )

        centre_indices: List[int] = []
        for i in feasible_coarse + lowest_disparity:
            if i not in centre_indices:
                centre_indices.append(i)
            if len(centre_indices) >= GRID_TOP_CENTRES:
                break

        offsets = np.arange(-GRID_FINE_RADIUS, GRID_FINE_RADIUS + 1)
        for centre_index in centre_indices:
            centre = np.array([
                rows[centre_index]["k_36"],
                rows[centre_index]["k_3_12"],
                rows[centre_index]["k_6_12"],
            ], dtype=float)
            axes = []
            for value in centre:
                axis = np.clip(
                    value + GRID_FINE_STEP * offsets,
                    FAIRNESS_STICK_MIN,
                    FAIRNESS_STICK_MAX,
                )
                axes.append(np.unique(np.round(axis, 7)))
            for stick_array in product(*axes):
                evaluate(stick_array, "fine")

    search_df = pd.DataFrame(rows)
    valid_indices = [
        i for i, row in enumerate(rows)
        if row["status"] in ACCEPTED_STATUSES
        and np.isfinite(row["total_DER_production_MW_scenarios"])
        and np.isfinite(row["average_utilisation_disparity"])
    ]
    if not valid_indices:
        return {
            "status": "all_grid_candidates_failed",
            "objective_value": np.nan,
        }, search_df

    feasible_indices = [
        i for i in valid_indices if bool(rows[i]["meets_target_disparity"])
    ]
    if feasible_indices:
        selected_index = max(feasible_indices, key=ranking_key)
        selection_rule = (
            "highest-production grid candidate satisfying the explicit "
            "average-utilisation-disparity constraint"
        )
    else:
        selected_index = min(
            valid_indices,
            key=lambda i: (
                float(rows[i]["average_utilisation_disparity"]),
                -float(rows[i]["total_DER_production_MW_scenarios"]),
            ),
        )
        selection_rule = (
            "minimum-disparity grid candidate because no evaluated point "
            "met the specified disparity limit"
        )

    selected_result = results[selected_index]
    selected_result["fairness_stick_target_disparity"] = float(target_disparity)
    selected_result["fairness_stick_selection_rule"] = selection_rule
    selected_result["fairness_stick_selected_index"] = int(selected_index)
    selected_result["fairness_stick_search_method"] = (
        "full-range Cartesian coarse grid with local fine refinement"
    )
    selected_result["grid_total_evaluations"] = len(rows)
    selected_result["grid_search_time_seconds"] = (
        time.perf_counter() - search_start
    )
    selected_result["grid_coarse_steps_per_dimension"] = number_of_steps
    selected_result["grid_fine_step"] = GRID_FINE_STEP
    selected_result["grid_fine_radius"] = GRID_FINE_RADIUS

    search_df["selected"] = False
    search_df.loc[selected_index, "selected"] = True
    return selected_result, search_df


def save_search_method_comparison_plots(
    baseline_result: Dict[str, object],
    grid_result: Dict[str, object],
    grid_df: pd.DataFrame,
    bo_result: Dict[str, object],
    bo_df: pd.DataFrame,
) -> None:
    """Create thesis-ready Grid Search versus BO comparison figures."""
    method_rows = [
        {
            "method": "Grid Search",
            "evaluations": int(grid_result["grid_total_evaluations"]),
            "time_seconds": float(grid_result["grid_search_time_seconds"]),
            "production": float(grid_result["total_der_production_MW_scenarios"]),
            "disparity": float(grid_result["achieved_average_utilisation_disparity"]),
        },
        {
            "method": "Bayesian Optimisation",
            "evaluations": int(bo_result["bayesian_total_evaluations"]),
            "time_seconds": float(bo_result["bayesian_search_time_seconds"]),
            "production": float(bo_result["total_der_production_MW_scenarios"]),
            "disparity": float(bo_result["achieved_average_utilisation_disparity"]),
        },
    ]
    methods = pd.DataFrame(method_rows)

    for column, ylabel, title, filename in [
        ("evaluations", "Number of OPF evaluations", "Search Evaluation Count", "grid_vs_bo_evaluation_count.png"),
        ("time_seconds", "Search time, s", "Search Calculation Time", "grid_vs_bo_calculation_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        bars = ax.bar(methods["method"], methods[column])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        for bar, value in zip(bars, methods[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.0f}" if column == "evaluations" else f"{value:.1f}",
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close(fig)

    baseline_prod = float(baseline_result["total_der_production_MW_scenarios"])
    baseline_disp = float(baseline_result["achieved_average_utilisation_disparity"])
    labels = ["Unconstrained", "Grid Search", "Bayesian Optimisation"]
    productions = [
        baseline_prod,
        float(grid_result["total_der_production_MW_scenarios"]),
        float(bo_result["total_der_production_MW_scenarios"]),
    ]
    disparities = [
        baseline_disp,
        float(grid_result["achieved_average_utilisation_disparity"]),
        float(bo_result["achieved_average_utilisation_disparity"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, productions)
    ax.set_ylabel("Total DER production, MW-scenarios")
    ax.set_title("Production Comparison")
    ax.grid(axis="y")
    for bar, value in zip(bars, productions):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unconstrained_grid_bo_production.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, disparities)
    ax.axhline(
        FAIRNESS_STICK_TARGET_DISPARITY,
        linestyle="--",
        label=f"Fairness limit = {FAIRNESS_STICK_TARGET_DISPARITY:.2f}",
    )
    ax.set_ylabel("Average utilisation disparity")
    ax.set_title("Fairness Comparison")
    ax.grid(axis="y")
    ax.legend()
    for bar, value in zip(bars, disparities):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unconstrained_grid_bo_disparity.png", dpi=300)
    plt.close(fig)

    # Best feasible production as a function of completed OPF evaluations.
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, data in [("Grid Search", grid_df), ("Bayesian Optimisation", bo_df)]:
        ordered = data.sort_values("candidate").copy()
        x = np.arange(1, len(ordered) + 1)
        y = ordered["best_feasible_production_so_far_MW_scenarios"]
        ax.plot(x, y, label=name)
    ax.set_xlabel("Completed OPF evaluations")
    ax.set_ylabel("Best feasible production, MW-scenarios")
    ax.set_title("Search Convergence Comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grid_vs_bo_convergence.png", dpi=300)
    plt.close(fig)

    # Production-disparity samples from both searches.
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, data, marker_style in [
        ("Grid Search", grid_df, "o"),
        ("Bayesian Optimisation", bo_df, "x"),
    ]:
        valid = data[
            data["status"].isin(ACCEPTED_STATUSES)
            & np.isfinite(data["total_DER_production_MW_scenarios"])
            & np.isfinite(data["average_utilisation_disparity"])
        ]
        ax.scatter(
            valid["average_utilisation_disparity"],
            valid["total_DER_production_MW_scenarios"],
            marker=marker_style,
            alpha=0.65,
            label=name,
        )
    ax.axvline(
        FAIRNESS_STICK_TARGET_DISPARITY,
        linestyle="--",
        label="Fairness limit",
    )
    ax.set_xlabel("Average utilisation disparity")
    ax.set_ylabel("Total DER production, MW-scenarios")
    ax.set_title("Production-Fairness Search Samples")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grid_vs_bo_production_disparity.png", dpi=300)
    plt.close(fig)

# ============================================================
# SOC DistFlow validation
# ============================================================

def solve_soc_distflow_loadflow(
    l_P: np.ndarray,
    l_Q: np.ndarray,
    p_der_fixed: np.ndarray,
    slack_voltage_pu: float,
) -> Dict[str, object]:
    """
    Fixed-injection SOC DistFlow validation.

    No Q support and no thermal constraints are imposed.
    """
    edge_count = len(edges)
    edge_index = {edge: idx for idx, edge in enumerate(edges)}

    P_edge = cp.Variable(edge_count, name="soc_P_branch")
    Q_edge = cp.Variable(edge_count, name="soc_Q_branch")
    L_edge = cp.Variable(
        edge_count,
        nonneg=True,
        name="soc_current_squared",
    )
    v_sq = cp.Variable(N, name="soc_voltage_squared")

    constraints = [
        v_sq[SLACK] == float(slack_voltage_pu) ** 2,
        v_sq >= 1e-8,
    ]

    for node in range(1, N):
        upstream = parent[node]
        edge_id = edge_index[(upstream, node)]

        if children[node]:
            downstream_p = cp.sum(
                cp.hstack([
                    P_edge[edge_index[(node, child)]]
                    for child in children[node]
                ])
            )
            downstream_q = cp.sum(
                cp.hstack([
                    Q_edge[edge_index[(node, child)]]
                    for child in children[node]
                ])
            )
        else:
            downstream_p = 0.0
            downstream_q = 0.0

        rij = float(r[upstream, node])
        xij = float(x[upstream, node])

        constraints.extend([
            P_edge[edge_id]
            == float(l_P[node] - p_der_fixed[node])
            + downstream_p
            + rij * L_edge[edge_id],

            Q_edge[edge_id]
            == float(l_Q[node])
            + downstream_q
            + xij * L_edge[edge_id],

            v_sq[node]
            == v_sq[upstream]
            - 2.0 * (
                rij * P_edge[edge_id]
                + xij * Q_edge[edge_id]
            )
            + (rij ** 2 + xij ** 2) * L_edge[edge_id],

            cp.SOC(
                v_sq[upstream] + L_edge[edge_id],
                cp.hstack([
                    2.0 * P_edge[edge_id],
                    2.0 * Q_edge[edge_id],
                    v_sq[upstream] - L_edge[edge_id],
                ]),
            ),
        ])

    losses = cp.sum(
        cp.hstack([
            float(r[i, j]) * L_edge[edge_index[(i, j)]]
            for i, j in edges
        ])
    )

    problem = cp.Problem(cp.Minimize(losses), constraints)
    solve_cvxpy_problem(problem)

    if problem.status not in ACCEPTED_STATUSES:
        return {
            "status": problem.status,
            "voltage": np.full(N, np.nan),
            "P": np.full((N, N), np.nan),
            "Q": np.full((N, N), np.nan),
            "L": np.full((N, N), np.nan),
            "losses_MW": np.nan,
            "max_abs_cone_gap": np.nan,
            "relaxation_fully_tight": False,
        }

    P_matrix = np.zeros((N, N))
    Q_matrix = np.zeros((N, N))
    L_matrix = np.zeros((N, N))
    v_sq_value = np.asarray(v_sq.value, dtype=float)

    cone_gaps = []

    for edge_id, (i, j) in enumerate(edges):
        P_matrix[i, j] = float(P_edge.value[edge_id])
        Q_matrix[i, j] = float(Q_edge.value[edge_id])
        L_matrix[i, j] = max(float(L_edge.value[edge_id]), 0.0)

        lhs = P_matrix[i, j] ** 2 + Q_matrix[i, j] ** 2
        rhs = v_sq_value[i] * L_matrix[i, j]
        cone_gaps.append(abs(rhs - lhs))

    max_abs_cone_gap = float(np.max(cone_gaps))

    return {
        "status": problem.status,
        "voltage": np.sqrt(np.maximum(v_sq_value, 0.0)),
        "P": P_matrix,
        "Q": Q_matrix,
        "L": L_matrix,
        "losses_MW": float(losses.value),
        "max_abs_cone_gap": max_abs_cone_gap,
        "relaxation_fully_tight": bool(
            max_abs_cone_gap <= SOC_TIGHTNESS_TOL
        ),
    }


def run_soc_validation(
    formulation_name: str,
    scenario_df: pd.DataFrame,
    joint_result: Dict[str, object],
) -> pd.DataFrame:
    rows = []

    if not RUN_SOC_VALIDATION:
        return pd.DataFrame(rows)

    for s, row in scenario_df.iterrows():
        soc = solve_soc_distflow_loadflow(
            l_P=joint_result["load_p"][s],
            l_Q=joint_result["load_q"][s],
            p_der_fixed=joint_result["p_der"][s],
            slack_voltage_pu=float(row["slack_voltage_pu"]),
        )

        linear_voltage = joint_result["voltage"][s]
        voltage_error = soc["voltage"] - linear_voltage

        rows.append({
            "formulation": formulation_name,
            "scenario": int(row["scenario"]),
            "slack_voltage_pu": float(row["slack_voltage_pu"]),
            "load_factor": float(row["load_factor"]),
            "soc_status": soc["status"],
            "linear_min_voltage_pu": float(
                np.nanmin(linear_voltage)
            ),
            "linear_max_voltage_pu": float(
                np.nanmax(linear_voltage)
            ),
            "soc_min_voltage_pu": float(
                np.nanmin(soc["voltage"])
            ),
            "soc_max_voltage_pu": float(
                np.nanmax(soc["voltage"])
            ),
            "mean_abs_voltage_error_pu": float(
                np.nanmean(np.abs(voltage_error))
            ),
            "max_abs_voltage_error_pu": float(
                np.nanmax(np.abs(voltage_error))
            ),
            "soc_upper_voltage_violation": bool(
                np.nanmax(soc["voltage"])
                > V_MAX + VOLTAGE_BINDING_TOL
            ),
            "soc_lower_voltage_violation": bool(
                np.nanmin(soc["voltage"])
                < V_MIN - VOLTAGE_BINDING_TOL
            ),
            "soc_losses_MW": soc["losses_MW"],
            "soc_max_abs_cone_gap": soc["max_abs_cone_gap"],
            "soc_relaxation_fully_tight": (
                soc["relaxation_fully_tight"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Post-processing
# ============================================================

def calculate_volt_watt_bound(
    pv_available: float,
    v_shift_squared: float,
    local_voltage_pu: float,
) -> float:
    raw_bound = (
        pv_available
        * (v_shift_squared - local_voltage_pu ** 2)
        / (V2_SQ - V1_SQ)
    )
    return float(raw_bound)


def create_result_tables(
    formulation_name: str,
    scenario_df: pd.DataFrame,
    result: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_rows = []
    node_rows = []
    branch_rows = []

    for s, row in scenario_df.iterrows():
        der_utilisation = (
            result["p_der"][s, DER_NODES]
            / result["pv_available"][s, DER_NODES]
        )

        scenario_rows.append({
            "formulation": formulation_name,
            "scenario": int(row["scenario"]),
            "slack_voltage_pu": float(row["slack_voltage_pu"]),
            "load_factor": float(row["load_factor"]),
            "total_DER_production_MW": float(
                np.sum(result["p_der"][s, DER_NODES])
            ),
            "mean_DER_utilisation": float(
                np.mean(der_utilisation)
            ),
            "absolute_utilisation_difference": float(
                np.max(der_utilisation) - np.min(der_utilisation)
            ),
            "minimum_voltage_pu": float(
                np.min(result["voltage"][s])
            ),
            "maximum_voltage_pu": float(
                np.max(result["voltage"][s])
            ),
            "upper_voltage_active_any": bool(
                np.any(
                    np.abs(
                        V_MAX - result["voltage"][s]
                    ) <= VOLTAGE_BINDING_TOL
                )
            ),
            "lower_voltage_active_any": bool(
                np.any(
                    np.abs(
                        result["voltage"][s] - V_MIN
                    ) <= VOLTAGE_BINDING_TOL
                )
            ),
        })

        for node in range(N):
            voltage = float(result["voltage"][s, node])
            available = float(result["pv_available"][s, node])
            p_value = float(result["p_der"][s, node])

            node_row = {
                "formulation": formulation_name,
                "scenario": int(row["scenario"]),
                "slack_voltage_pu": float(row["slack_voltage_pu"]),
                "load_factor": float(row["load_factor"]),
                "node": node,
                "voltage_pu": voltage,
                "upper_voltage_margin_pu": V_MAX - voltage,
                "lower_voltage_margin_pu": voltage - V_MIN,
                "upper_voltage_active": bool(
                    abs(V_MAX - voltage) <= VOLTAGE_BINDING_TOL
                ),
                "lower_voltage_active": bool(
                    abs(voltage - V_MIN) <= VOLTAGE_BINDING_TOL
                ),
                "load_P_MW": float(result["load_p"][s, node]),
                "load_Q_MVAr": float(result["load_q"][s, node]),
                "P_DER_MW": p_value,
                "PV_available_MW": available,
            }

            if node in DER_NODES:
                vw_bound = calculate_volt_watt_bound(
                    available,
                    result["v_shift_squared"][node],
                    voltage,
                )
                node_row.update({
                    "V_shift_pu": float(
                        result["v_shift_pu"][node]
                    ),
                    "utilisation": p_value / available,
                    "curtailment_MW": available - p_value,
                    "capacity_margin_MW": available - p_value,
                    "capacity_active": bool(
                        abs(available - p_value)
                        <= CAPACITY_BINDING_TOL
                    ),
                    "volt_watt_bound_MW": vw_bound,
                    "volt_watt_margin_MW": vw_bound - p_value,
                    "volt_watt_active": bool(
                        abs(vw_bound - p_value)
                        <= VOLT_WATT_BINDING_TOL
                    ),
                })
            else:
                node_row.update({
                    "V_shift_pu": np.nan,
                    "utilisation": np.nan,
                    "curtailment_MW": np.nan,
                    "capacity_margin_MW": np.nan,
                    "capacity_active": False,
                    "volt_watt_bound_MW": np.nan,
                    "volt_watt_margin_MW": np.nan,
                    "volt_watt_active": False,
                })

            node_rows.append(node_row)

        for i, j in edges:
            p_flow = float(result["P"][s, i, j])
            q_flow = float(result["Q"][s, i, j])
            voltage_drop = float(
                result["voltage"][s, i]
                - result["voltage"][s, j]
            )

            # Equality residuals are numerical checks, not binding tests.
            if children[j]:
                downstream_p = sum(
                    result["P"][s, j, child]
                    for child in children[j]
                )
                downstream_q = sum(
                    result["Q"][s, j, child]
                    for child in children[j]
                )
            else:
                downstream_p = 0.0
                downstream_q = 0.0

            p_residual = (
                p_flow
                - (
                    result["load_p"][s, j]
                    - result["p_der"][s, j]
                    + downstream_p
                )
            )
            q_residual = (
                q_flow
                - (
                    result["load_q"][s, j]
                    + downstream_q
                )
            )
            voltage_residual = (
                result["v_squared"][s, j]
                - (
                    result["v_squared"][s, i]
                    - 2.0 * (
                        r[i, j] * p_flow
                        + x[i, j] * q_flow
                    )
                )
            )

            branch_rows.append({
                "formulation": formulation_name,
                "scenario": int(row["scenario"]),
                "slack_voltage_pu": float(row["slack_voltage_pu"]),
                "load_factor": float(row["load_factor"]),
                "from_node": i,
                "to_node": j,
                "P_MW": p_flow,
                "Q_MVAr": q_flow,
                "apparent_power_MVA": float(
                    np.hypot(p_flow, q_flow)
                ),
                "sending_voltage_pu": float(
                    result["voltage"][s, i]
                ),
                "receiving_voltage_pu": float(
                    result["voltage"][s, j]
                ),
                "signed_voltage_difference_pu": voltage_drop,
                "absolute_voltage_difference_pu": abs(voltage_drop),
                "active_power_balance_residual_MW": float(
                    p_residual
                ),
                "reactive_power_balance_residual_MVAr": float(
                    q_residual
                ),
                "voltage_drop_residual_squared": float(
                    voltage_residual
                ),
            })

    return (
        pd.DataFrame(scenario_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(branch_rows),
    )


def build_summary(
    formulation_name: str,
    result: Dict[str, object],
    scenario_results: pd.DataFrame,
    node_results: pd.DataFrame,
    soc_results: pd.DataFrame,
) -> pd.DataFrame:
    der_results = node_results[
        node_results["node"].isin(DER_NODES)
    ].copy()

    summary = {
        "formulation": formulation_name,
        "status": result["status"],
        "number_of_scenarios": len(scenario_results),
        "total_DER_production_MW_scenarios": (
            scenario_results["total_DER_production_MW"].sum()
        ),
        "mean_total_DER_production_MW": (
            scenario_results["total_DER_production_MW"].mean()
        ),
        "mean_scenario_utilisation_disparity": (
            scenario_results[
                "absolute_utilisation_difference"
            ].mean()
        ),
        "average_utilisation_disparity_across_DERs": (
            result["achieved_average_utilisation_disparity"]
        ),
        "fairness_stick_k_36": result.get("k_36", np.nan),
        "fairness_stick_k_3_12": result.get("k_3_12", np.nan),
        "fairness_stick_k_6_12": result.get("k_6_12", np.nan),
        "fairness_stick_target_disparity": result.get(
            "fairness_stick_target_disparity",
            np.nan,
        ),
        "fairness_stick_selection_rule": result.get(
            "fairness_stick_selection_rule",
            "",
        ),
        "upper_voltage_active_scenario_count": int(
            scenario_results["upper_voltage_active_any"].sum()
        ),
        "lower_voltage_active_scenario_count": int(
            scenario_results["lower_voltage_active_any"].sum()
        ),
    }

    for node in DER_NODES:
        node_data = der_results[der_results["node"] == node]
        summary.update({
            f"V_shift_node_{node}_pu": result["v_shift_pu"][node],
            f"mean_node_{node}_utilisation": (
                node_data["utilisation"].mean()
            ),
            f"total_node_{node}_curtailment_MW_scenarios": (
                node_data["curtailment_MW"].sum()
            ),
            f"node_{node}_capacity_active_fraction": (
                node_data["capacity_active"].mean()
            ),
            f"node_{node}_volt_watt_active_fraction": (
                node_data["volt_watt_active"].mean()
            ),
        })

    if not soc_results.empty:
        summary.update({
            "SOC_solved_scenarios": int(
                soc_results["soc_status"]
                .isin(ACCEPTED_STATUSES)
                .sum()
            ),
            "SOC_upper_voltage_violation_scenarios": int(
                soc_results[
                    "soc_upper_voltage_violation"
                ].sum()
            ),
            "SOC_lower_voltage_violation_scenarios": int(
                soc_results[
                    "soc_lower_voltage_violation"
                ].sum()
            ),
            "mean_SOC_max_abs_voltage_error_pu": (
                soc_results[
                    "max_abs_voltage_error_pu"
                ].mean()
            ),
            "overall_SOC_max_abs_voltage_error_pu": (
                soc_results[
                    "max_abs_voltage_error_pu"
                ].max()
            ),
            "SOC_relaxation_tight_fraction": (
                soc_results[
                    "soc_relaxation_fully_tight"
                ].mean()
            ),
        })

    return pd.DataFrame([summary])


# ============================================================
# Plotting
# ============================================================

def save_heatmap(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    colourbar_label: str,
    filepath: Path,
) -> None:
    pivot = data.pivot_table(
        index="load_factor",
        columns="slack_voltage_pu",
        values=value_column,
        aggfunc="mean",
    )
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        ["{:.3f}".format(value) for value in pivot.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        ["{:.2f}".format(value) for value in pivot.index]
    )
    ax.set_xlabel("Slack-bus voltage, p.u.")
    ax.set_ylabel("Load factor")
    ax.set_title(title)

    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label(colourbar_label)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def save_formulation_plots(
    formulation_name: str,
    result: Dict[str, object],
    scenario_results: pd.DataFrame,
    node_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    safe_name = (
        formulation_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    der_results = node_results[
        node_results["node"].isin(DER_NODES)
    ].copy()

    # Fixed Volt-Watt curves and scenario operating points.
    voltage_axis = np.linspace(1.0, 1.055, 500)

    fig, ax = plt.subplots(figsize=(9, 6))
    for node in DER_NODES:
        normalised_limit = np.clip(
            (
                result["v_shift_squared"][node]
                - voltage_axis ** 2
            )
            / (V2_SQ - V1_SQ),
            0.0,
            1.0,
        )
        ax.plot(
            voltage_axis,
            normalised_limit,
            label=(
                "Node {} curve, V_shift={:.4f} p.u."
                .format(node, result["v_shift_pu"][node])
            ),
        )

        node_data = der_results[
            der_results["node"] == node
        ]
        ax.scatter(
            node_data["voltage_pu"],
            node_data["utilisation"],
            alpha=0.65,
            label="Node {} operating points".format(node),
        )

    ax.axvline(
        V_MAX,
        linestyle="--",
        label="Voltage upper limit",
    )
    ax.set_xlabel("Local voltage, p.u.")
    ax.set_ylabel("Normalised DER production")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "Fixed Volt-Watt Curves and Operating Points: "
        + formulation_name
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / (
            safe_name
            + "_fixed_volt_watt_curves_and_points.png"
        ),
        dpi=300,
    )
    plt.close(fig)

    save_heatmap(
        scenario_results,
        "total_DER_production_MW",
        "Total DER Production: " + formulation_name,
        "Total DER production, MW",
        output_dir / (
            safe_name + "_total_DER_production_heatmap.png"
        ),
    )

    save_heatmap(
        scenario_results,
        "absolute_utilisation_difference",
        "DER Utilisation Difference: " + formulation_name,
        "Absolute utilisation difference",
        output_dir / (
            safe_name + "_utilisation_difference_heatmap.png"
        ),
    )

    for node in DER_NODES:
        node_data = der_results[
            der_results["node"] == node
        ].copy()

        save_heatmap(
            node_data,
            "utilisation",
            "DER Utilisation at Node {}: {}".format(
                node,
                formulation_name,
            ),
            "DER utilisation",
            output_dir / (
                safe_name
                + "_node_{}_utilisation_heatmap.png".format(node)
            ),
        )

        save_heatmap(
            node_data,
            "curtailment_MW",
            "DER Curtailment at Node {}: {}".format(
                node,
                formulation_name,
            ),
            "Curtailment, MW",
            output_dir / (
                safe_name
                + "_node_{}_curtailment_heatmap.png".format(node)
            ),
        )

    # Constraint activity by scenario.
    fig, ax = plt.subplots(figsize=(9, 6))
    for node in DER_NODES:
        node_data = der_results[
            der_results["node"] == node
        ]
        ax.scatter(
            node_data["scenario"],
            node_data["volt_watt_margin_MW"],
            alpha=0.65,
            label="Node {}".format(node),
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Volt-Watt upper-bound margin, MW")
    ax.set_title(
        "Volt-Watt Constraint Margin: " + formulation_name
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / (
            safe_name + "_volt_watt_constraint_margin.png"
        ),
        dpi=300,
    )
    plt.close(fig)




def save_fairness_stick_sweep_plots(
    sweep_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save diagnostics for the three-element pairwise fairness-stick search.
    """
    valid = sweep_df[
        sweep_df["status"].isin(ACCEPTED_STATUSES)
    ].copy()

    if valid.empty:
        return

    valid = valid.sort_values("candidate")
    selected = valid[valid["selected"]]

    # Candidate index versus total production.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        valid["candidate"],
        valid["total_DER_production_MW_scenarios"],
        s=25,
        alpha=0.65,
        label="Evaluated coefficient arrays",
    )
    if not selected.empty:
        ax.scatter(
            selected["candidate"],
            selected["total_DER_production_MW_scenarios"],
            s=140,
            marker="*",
            label="Selected array",
            zorder=5,
        )
    ax.set_xlabel("Bayesian evaluation")
    ax.set_ylabel("Total DER production across scenarios, MW")
    ax.set_title("Bayesian Search: DER Production")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "pairwise_stick_candidate_vs_production.png",
        dpi=300,
    )
    plt.close(fig)

    # Candidate index versus utilisation disparity.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        valid["candidate"],
        valid["average_utilisation_disparity"],
        s=25,
        alpha=0.65,
        label="Evaluated coefficient arrays",
    )
    ax.axhline(
        FAIRNESS_STICK_TARGET_DISPARITY,
        linestyle="--",
        label=(
            "Target disparity = "
            f"{FAIRNESS_STICK_TARGET_DISPARITY:.3f}"
        ),
    )
    if not selected.empty:
        ax.scatter(
            selected["candidate"],
            selected["average_utilisation_disparity"],
            s=140,
            marker="*",
            label="Selected array",
            zorder=5,
        )
    ax.set_xlabel("Bayesian evaluation")
    ax.set_ylabel("Average utilisation disparity")
    ax.set_title("Bayesian Search: Utilisation Disparity")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "pairwise_stick_candidate_vs_disparity.png",
        dpi=300,
    )
    plt.close(fig)

    # Three coefficients across all candidates.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(valid["candidate"], valid["k_36"], label=r"$k_{3,6}$")
    ax.plot(valid["candidate"], valid["k_3_12"], label=r"$k_{3,12}$")
    ax.plot(valid["candidate"], valid["k_6_12"], label=r"$k_{6,12}$")
    ax.set_xlabel("Bayesian evaluation")
    ax.set_ylabel("Fairness-stick coefficient")
    ax.set_title("Three Pairwise Fairness-Stick Coefficients")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "pairwise_fairness_stick_coefficients.png",
        dpi=300,
    )
    plt.close(fig)


# ============================================================
# Complete workflow
# ============================================================

def run_formulation(
    formulation_name: str,
    scenario_df: pd.DataFrame,
    result: Dict[str, object],
) -> Dict[str, pd.DataFrame]:
    if result["status"] not in ACCEPTED_STATUSES:
        raise RuntimeError(
            formulation_name
            + " failed with status "
            + str(result["status"])
        )

    safe_name = (
        formulation_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    output_dir = OUTPUT_DIR / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_results, node_results, branch_results = (
        create_result_tables(
            formulation_name,
            scenario_df,
            result,
        )
    )

    soc_results = run_soc_validation(
        formulation_name,
        scenario_df,
        result,
    )

    summary = build_summary(
        formulation_name,
        result,
        scenario_results,
        node_results,
        soc_results,
    )

    shift_results = pd.DataFrame([
        {
            "formulation": formulation_name,
            "DER_node": node,
            "V_shift_squared": (
                result["v_shift_squared"][node]
            ),
            "V_shift_pu": result["v_shift_pu"][node],
        }
        for node in DER_NODES
    ])

    scenario_results.to_csv(
        output_dir / "scenario_results.csv",
        index=False,
    )
    node_results.to_csv(
        output_dir / "node_and_constraint_results.csv",
        index=False,
    )
    branch_results.to_csv(
        output_dir / "branch_DistFlow_diagnostics.csv",
        index=False,
    )
    soc_results.to_csv(
        output_dir / "SOC_validation_results.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    shift_results.to_csv(
        output_dir / "optimised_V_shift.csv",
        index=False,
    )

    save_formulation_plots(
        formulation_name,
        result,
        scenario_results,
        node_results,
        output_dir,
    )

    print("\n" + "=" * 80)
    print(formulation_name)
    print("=" * 80)
    print(shift_results.to_string(index=False))
    print(summary.to_string(index=False))

    return {
        "scenario_results": scenario_results,
        "node_results": node_results,
        "branch_results": branch_results,
        "soc_results": soc_results,
        "summary": summary,
        "shift_results": shift_results,
    }


def main() -> None:
    scenario_df = create_probabilistic_scenarios()
    scenario_df.to_csv(OUTPUT_DIR / "final_correlated_nodal_load_operating_scenarios.csv", index=False)
    save_scenario_sampling_plots(scenario_df, OUTPUT_DIR)

    settings = pd.DataFrame([{
        "number_of_scenarios": len(scenario_df),
        "scenario_generation": "separate correlated nodal loads with negatively correlated slack voltage",
        "scenario_distribution": SCENARIO_DISTRIBUTION,
        "scenario_random_seed": SCENARIO_RANDOM_SEED,
        "target_pairwise_nodal_load_correlation": TARGET_NODAL_LOAD_CORRELATION,
        "target_slack_aggregate_load_correlation": TARGET_SLACK_AGGREGATE_LOAD_CORRELATION,
        "calibrated_latent_slack_load_correlation": float(scenario_df["calibrated_latent_slack_load_correlation"].iloc[0]),
        "slack_correlation_calibration_tolerance": SLACK_CORRELATION_CALIBRATION_TOLERANCE,
        "realised_mean_pairwise_nodal_load_correlation": float(
            scenario_df["realised_mean_pairwise_load_correlation"].iloc[0]
        ),
        "realised_pearson_slack_aggregate_load_correlation": float(
            scenario_df["realised_pearson_slack_aggregate_load_correlation"].iloc[0]
        ),
        "realised_spearman_slack_aggregate_load_correlation": float(
            scenario_df["realised_spearman_slack_aggregate_load_correlation"].iloc[0]
        ),
        "slack_voltage_min_pu": SLACK_VOLTAGE_MIN,
        "slack_voltage_max_pu": SLACK_VOLTAGE_MAX,
        "slack_voltage_mean_pu": SLACK_VOLTAGE_MEAN,
        "slack_voltage_std_pu": SLACK_VOLTAGE_STD,
        "load_factor_min": LOAD_FACTOR_MIN,
        "load_factor_max": LOAD_FACTOR_MAX,
        "load_factor_mean": LOAD_FACTOR_MEAN,
        "load_factor_std": LOAD_FACTOR_STD,
        "DER_nodes": ",".join(str(x) for x in DER_NODES),
        "PV_capacity_each_MW": PV_CAPACITY_PER_DER_MW,
        "base_load_multiplier": 8.0,
        "impedance_multiplier_after_swap": 0.075,
        "r_x_swapped": True,
        "reactive_power_control": False,
        "thermal_limits": False,
        "one_shared_shift_per_DER_across_scenarios": True,
        "V_shift_lower_bound_pu": np.sqrt(VSHIFT_MIN_SQ),
        "V_shift_upper_bound_pu": np.sqrt(VSHIFT_MAX_SQ),
        "fairness_stick_min": FAIRNESS_STICK_MIN,
        "fairness_stick_max": FAIRNESS_STICK_MAX,
        "target_disparity": FAIRNESS_STICK_TARGET_DISPARITY,
        "grid_search_enabled": RUN_GRID_SEARCH,
        "grid_coarse_steps": GRID_COARSE_STEPS,
        "grid_fine_step": GRID_FINE_STEP,
        "grid_fine_radius": GRID_FINE_RADIUS,
        "bayesian_initial_evaluations": BO_INITIAL_EVALUATIONS,
        "bayesian_iterations": BO_ITERATIONS,
        "bayesian_library": "scikit-optimize",
        "bayesian_acquisition": "Expected Improvement with fairness penalty wrapper",
        "bayesian_random_seed": BO_RANDOM_SEED,
        "SOC_validation_enabled": RUN_SOC_VALIDATION,
    }])
    settings.to_csv(OUTPUT_DIR / "study_settings.csv", index=False)

    print("Output directory:", OUTPUT_DIR)
    print("Number of scenarios:", len(scenario_df))
    print("Scenario generation: separate correlated nodal loads and negatively correlated slack voltage")
    print("Scenario distribution:", SCENARIO_DISTRIBUTION)
    print("Scenario random seed:", SCENARIO_RANDOM_SEED)
    print("Target pairwise nodal-load correlation:", TARGET_NODAL_LOAD_CORRELATION)
    print(
        "Realised mean pairwise nodal-load correlation:",
        f"{scenario_df['realised_mean_pairwise_load_correlation'].iloc[0]:.6f}",
    )
    print("Target slack-aggregate-load correlation:", TARGET_SLACK_AGGREGATE_LOAD_CORRELATION)
    print(
        "Calibrated latent slack-load correlation:",
        f"{scenario_df['calibrated_latent_slack_load_correlation'].iloc[0]:.6f}",
    )
    print(
        "Calibration absolute error:",
        f"{scenario_df['calibration_absolute_error'].iloc[0]:.6f}",
    )
    print(
        "Realised Pearson slack-aggregate-load correlation:",
        f"{scenario_df['realised_pearson_slack_aggregate_load_correlation'].iloc[0]:.6f}",
    )
    print(
        "Realised Spearman slack-aggregate-load correlation:",
        f"{scenario_df['realised_spearman_slack_aggregate_load_correlation'].iloc[0]:.6f}",
    )
    print(
        "Slack-voltage range:",
        SLACK_VOLTAGE_MIN,
        "to",
        SLACK_VOLTAGE_MAX,
        "p.u.",
    )
    print("Load-factor range:", LOAD_FACTOR_MIN, "to", LOAD_FACTOR_MAX)
    print("DER nodes:", DER_NODES)
    print("DER capacity at each participating node:", PV_CAPACITY_PER_DER_MW, "MW")
    print("Grid-search comparison enabled:", RUN_GRID_SEARCH)
    print("Target average utilisation disparity:", FAIRNESS_STICK_TARGET_DISPARITY)
    print("Both searches use the same solve_single_fairness_stick OPF wrapper.")

    # Case 1: three DERs without fairness, maximising total export.
    print("\nSolving unconstrained total-export benchmark...")
    baseline_start = time.perf_counter()
    baseline_result = solve_utilisation_oriented_model(scenario_df)
    baseline_time = time.perf_counter() - baseline_start
    baseline_outputs = run_formulation(
        "Unconstrained Volt-Watt Optimisation",
        scenario_df,
        baseline_result,
    )

    # Case 2a: previous grid search over the three fairness coefficients.
    grid_result = None
    grid_df = pd.DataFrame()
    grid_outputs = None
    if RUN_GRID_SEARCH:
        print("\nSolving fairness-constrained coarse-to-fine grid search...")
        grid_result, grid_df = solve_grid_search_fairness_stick_model(
            scenario_df=scenario_df,
            target_disparity=FAIRNESS_STICK_TARGET_DISPARITY,
            number_of_steps=GRID_COARSE_STEPS,
        )
        grid_df.to_csv(OUTPUT_DIR / "grid_search_evaluations.csv", index=False)
        grid_outputs = run_formulation(
            "Fairness-Constrained Grid Search",
            scenario_df,
            grid_result,
        )

    # Case 2b: Bayesian optimisation over exactly the same coefficients and OPF.
    print("\nSolving fairness-constrained Bayesian optimisation...")
    bo_result, bo_df = solve_fairness_stick_model(
        scenario_df=scenario_df,
        target_disparity=FAIRNESS_STICK_TARGET_DISPARITY,
        n_initial=BO_INITIAL_EVALUATIONS,
        n_iterations=BO_ITERATIONS,
    )
    bo_df.to_csv(OUTPUT_DIR / "bayesian_optimization_evaluations.csv", index=False)
    save_fairness_stick_sweep_plots(bo_df, OUTPUT_DIR)
    bo_outputs = run_formulation(
        "Fairness-Constrained Bayesian Optimisation",
        scenario_df,
        bo_result,
    )

    # Case-level comparison.
    summary_frames = [baseline_outputs["summary"]]
    if grid_outputs is not None:
        summary_frames.append(grid_outputs["summary"])
    summary_frames.append(bo_outputs["summary"])
    pd.concat(summary_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_case_summary.csv",
        index=False,
    )

    baseline_prod = float(baseline_result["total_der_production_MW_scenarios"])
    baseline_disp = float(baseline_result["achieved_average_utilisation_disparity"])
    comparison_rows = [{
        "method": "Unconstrained benchmark",
        "search_method": "Direct OPF",
        "OPF_parameter_evaluations": 1,
        "search_time_seconds": baseline_time,
        "total_DER_production_MW_scenarios": baseline_prod,
        "average_utilisation_disparity": baseline_disp,
        "meets_5pct_disparity_limit": baseline_disp <= FAIRNESS_STICK_TARGET_DISPARITY + 1e-8,
        "k_36": np.nan,
        "k_3_12": np.nan,
        "k_6_12": np.nan,
        "production_loss_vs_unconstrained_MW_scenarios": 0.0,
        "production_loss_vs_unconstrained_pct": 0.0,
        "disparity_reduction_vs_unconstrained_pct": 0.0,
    }]

    if grid_result is not None:
        grid_prod = float(grid_result["total_der_production_MW_scenarios"])
        grid_disp = float(grid_result["achieved_average_utilisation_disparity"])
        comparison_rows.append({
            "method": "Fairness-constrained Grid Search",
            "search_method": "Coarse-to-fine Cartesian grid",
            "OPF_parameter_evaluations": int(grid_result["grid_total_evaluations"]),
            "search_time_seconds": float(grid_result["grid_search_time_seconds"]),
            "total_DER_production_MW_scenarios": grid_prod,
            "average_utilisation_disparity": grid_disp,
            "meets_5pct_disparity_limit": grid_disp <= FAIRNESS_STICK_TARGET_DISPARITY + 1e-8,
            "k_36": float(grid_result["k_36"]),
            "k_3_12": float(grid_result["k_3_12"]),
            "k_6_12": float(grid_result["k_6_12"]),
            "production_loss_vs_unconstrained_MW_scenarios": baseline_prod - grid_prod,
            "production_loss_vs_unconstrained_pct": 100.0 * (baseline_prod - grid_prod) / baseline_prod,
            "disparity_reduction_vs_unconstrained_pct": 100.0 * (baseline_disp - grid_disp) / baseline_disp,
        })

    bo_prod = float(bo_result["total_der_production_MW_scenarios"])
    bo_disp = float(bo_result["achieved_average_utilisation_disparity"])
    comparison_rows.append({
        "method": "Fairness-constrained Bayesian Optimisation",
        "search_method": "Gaussian-process BO with Expected Improvement",
        "OPF_parameter_evaluations": int(bo_result["bayesian_total_evaluations"]),
        "search_time_seconds": float(bo_result["bayesian_search_time_seconds"]),
        "total_DER_production_MW_scenarios": bo_prod,
        "average_utilisation_disparity": bo_disp,
        "meets_5pct_disparity_limit": bo_disp <= FAIRNESS_STICK_TARGET_DISPARITY + 1e-8,
        "k_36": float(bo_result["k_36"]),
        "k_3_12": float(bo_result["k_3_12"]),
        "k_6_12": float(bo_result["k_6_12"]),
        "production_loss_vs_unconstrained_MW_scenarios": baseline_prod - bo_prod,
        "production_loss_vs_unconstrained_pct": 100.0 * (baseline_prod - bo_prod) / baseline_prod,
        "disparity_reduction_vs_unconstrained_pct": 100.0 * (baseline_disp - bo_disp) / baseline_disp,
    })

    comparison = pd.DataFrame(comparison_rows)
    if grid_result is not None:
        grid_eval = int(grid_result["grid_total_evaluations"])
        bo_eval = int(bo_result["bayesian_total_evaluations"])
        grid_time = float(grid_result["grid_search_time_seconds"])
        bo_time = float(bo_result["bayesian_search_time_seconds"])
        comparison["BO_evaluation_reduction_vs_grid_pct"] = np.nan
        comparison["BO_time_reduction_vs_grid_pct"] = np.nan
        comparison["BO_production_gain_vs_grid_MW_scenarios"] = np.nan
        bo_mask = comparison["method"].str.contains("Bayesian")
        comparison.loc[bo_mask, "BO_evaluation_reduction_vs_grid_pct"] = (
            100.0 * (grid_eval - bo_eval) / grid_eval
        )
        comparison.loc[bo_mask, "BO_time_reduction_vs_grid_pct"] = (
            100.0 * (grid_time - bo_time) / grid_time
            if grid_time > 0.0 else np.nan
        )
        comparison.loc[bo_mask, "BO_production_gain_vs_grid_MW_scenarios"] = (
            bo_prod - float(grid_result["total_der_production_MW_scenarios"])
        )

    comparison.to_csv(OUTPUT_DIR / "grid_vs_bayesian_method_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("Final method comparison")
    print("=" * 80)
    print(comparison.to_string(index=False))

    if grid_result is not None:
        save_search_method_comparison_plots(
            baseline_result,
            grid_result,
            grid_df,
            bo_result,
            bo_df,
        )

    print("\nFinished.")
    print("Results saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
