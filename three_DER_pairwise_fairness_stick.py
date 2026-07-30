"""
the three coefficients describe the possible pairwise relationships
between DER nodes 3, 6 and 12:

    V_shift_sq[3] <= k_36   * V_shift_sq[6]
    V_shift_sq[3] <= k_3_12 * V_shift_sq[12]
    V_shift_sq[6] <= k_6_12 * V_shift_sq[12]

The three coefficients are searched independently. Each inner problem remains
a continuous convex optimisation problem because the coefficients are fixed
parameters during each solve and V_shift remains an optimisation variable.

The outer search selects the highest-production candidate satisfying the
target average utilisation disparity.
"""


import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import product

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Study settings
# ============================================================

OUTPUT_DIR = Path("results_three_DER_pairwise_fairness_stick_fast_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 13
SLACK = 0
DER_NODES = [3, 6, 12]

# Keep the original 10 x 10 scenario grid.
SLACK_VOLTAGE_MIN = 0.98
SLACK_VOLTAGE_MAX = 1.03
N_SLACK_STEPS = int(os.environ.get("OE_N_SLACK_STEPS", "10"))

LOAD_FACTOR_MIN = 0.20
LOAD_FACTOR_MAX = 0.80
N_LOAD_STEPS = int(os.environ.get("OE_N_LOAD_STEPS", "10"))

PV_CAPACITY_PER_DER_MW = 10.0

# Maximum allowed difference between the average utilisations of any
# two participating DERs. The default value 0.05 means 5 percentage points.
AVERAGE_UTILISATION_DISPARITY_LIMIT = float(
    os.environ.get("OE_FAIRNESS_EPSILON", "0.05")
)


# k = 1.0 imposes only ordering/equality limits and is closest to the
# utilisation-oriented setting. Smaller k imposes stronger separation.
FAIRNESS_STICK_MAX = float(
    os.environ.get("OE_FAIRNESS_STICK_MAX", "1.0")
)
FAIRNESS_STICK_MIN = float(
    os.environ.get("OE_FAIRNESS_STICK_MIN", "0.97")
)
# Coarse-to-fine parameter-search controls.
# Seven coarse values cover the complete interval [0.97, 1.00], producing
# 7^3 = 343 coarse coefficient arrays. The two most promising regions are then
# Only the best coarse region is refined using a compact 3 x 3 x 3 grid.
# The ultra-fine stage is disabled by default to minimise runtime.
FAIRNESS_STICK_COARSE_STEPS = int(
    os.environ.get("OE_FAIRNESS_STICK_COARSE_STEPS", "4")
)
FAIRNESS_STICK_TOP_CENTRES = int(
    os.environ.get("OE_FAIRNESS_STICK_TOP_CENTRES", "1")
)
FAIRNESS_STICK_FINE_STEP = float(
    os.environ.get("OE_FAIRNESS_STICK_FINE_STEP", "0.001")
)
FAIRNESS_STICK_FINE_RADIUS = int(
    os.environ.get("OE_FAIRNESS_STICK_FINE_RADIUS", "1")
)
FAIRNESS_STICK_ULTRA_FINE_STEP = float(
    os.environ.get("OE_FAIRNESS_STICK_ULTRA_FINE_STEP", "0.0001")
)
FAIRNESS_STICK_ULTRA_FINE_RADIUS = int(
    os.environ.get("OE_FAIRNESS_STICK_ULTRA_FINE_RADIUS", "3")
)
RUN_ULTRA_FINE_SEARCH = (
    os.environ.get("OE_RUN_ULTRA_FINE_SEARCH", "0") == "1"
)
FAIRNESS_STICK_TARGET_DISPARITY = float(
    os.environ.get("OE_FAIRNESS_STICK_TARGET_DISPARITY", "0.05")
)

V_MIN = 0.95
V_MAX = 1.05

# squared-voltage Volt-Watt formulation.
V1_SQ = 1.04 ** 2
V2_SQ = V_MAX ** 2
VSHIFT_MIN_SQ = 1.01 ** 2
VSHIFT_MAX_SQ = 1.05 ** 2

ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")

# Diagnostics.
VOLTAGE_BINDING_TOL = 1e-4
CAPACITY_BINDING_TOL = 1e-5
VOLT_WATT_BINDING_TOL = 1e-5
FAIRNESS_BINDING_TOL = 1e-5
SOC_TIGHTNESS_TOL = 1e-5
FAIRNESS_OBJECTIVE_TOL = 1e-6

# Optional run controls for development.
RUN_SOC_VALIDATION = os.environ.get("OE_RUN_SOC", "1") == "1"
SOLVER_VERBOSE = os.environ.get("OE_SOLVER_VERBOSE", "0") == "1"


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

I_max = np.array([
[0, 3.0441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1.4178, 0, 0.9591, 0, 3.0441, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 3.1275, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.9591, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 3.0441, 3.1275, 0, 0.9591, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.37193, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9591, 1.2927],
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

# Preserve originals for traceability, then swap r and x.
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

def create_scenario_grid() -> pd.DataFrame:
    rows = []
    scenario = 0

    slack_values = np.linspace(
        SLACK_VOLTAGE_MIN,
        SLACK_VOLTAGE_MAX,
        N_SLACK_STEPS,
    )
    load_values = np.linspace(
        LOAD_FACTOR_MIN,
        LOAD_FACTOR_MAX,
        N_LOAD_STEPS,
    )

    for slack_voltage_pu in slack_values:
        for load_factor in load_values:
            rows.append({
                "scenario": scenario,
                "slack_voltage_pu": float(slack_voltage_pu),
                "load_factor": float(load_factor),
            })
            scenario += 1

    return pd.DataFrame(rows)


def build_scenario_arrays(
    scenario_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario_count = len(scenario_df)

    load_p = np.zeros((scenario_count, N))
    load_q = np.zeros((scenario_count, N))
    pv_available = np.zeros((scenario_count, N))

    for s, row in scenario_df.iterrows():
        load_factor = float(row["load_factor"])
        load_p[s, :] = base_l_P * load_factor
        load_q[s, :] = base_l_Q * load_factor
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

def build_joint_local_envelope_problem(
    scenario_df: pd.DataFrame,
    formulation: str,
    fairness_sum_target: Optional[float] = None,
    average_utilisation_disparity_limit: Optional[float] = None,
    fairness_stick_array: Optional[Sequence[float]] = None,
) -> Tuple[cp.Problem, Dict[str, object]]:
    """
    Construct one optimisation problem containing every operating scenario.

    formulation:
        utilisation
        fairness_stage_1
        fairness_stage_2
        disparity_constrained
        fairness_stick
    """
    valid_formulations = {
        "utilisation",
        "fairness_stage_1",
        "fairness_stage_2",
        "disparity_constrained",
        "fairness_stick",
    }
    if formulation not in valid_formulations:
        raise ValueError("Unknown formulation: " + formulation)

    scenario_count = len(scenario_df)
    load_p, load_q, pv_available = build_scenario_arrays(scenario_df)

    # Scenario-independent control parameter.
    v_shift_sq = cp.Variable(N, name="shared_v_shift_squared")

    # Scenario-dependent operating variables.
    p_der = cp.Variable(
        (scenario_count, N),
        nonneg=True,
        name="scenario_DER_active_power",
    )
    v_sq = cp.Variable(
        (scenario_count, N),
        name="scenario_voltage_squared",
    )

    # CVXPY does not support a native 3-D variable in all older versions.
    # A list of matrices is therefore used for branch flows.
    P = [
        cp.Variable((N, N), name="P_s{}".format(s))
        for s in range(scenario_count)
    ]
    Q = [
        cp.Variable((N, N), name="Q_s{}".format(s))
        for s in range(scenario_count)
    ]

    fairness_enabled = formulation in {"fairness_stage_1", "fairness_stage_2"}
    alpha = None
    if fairness_enabled:
        # One common DER participation coefficient in each scenario.
        # It is common across all participating DER nodes, while varying by scenario.
        alpha = cp.Variable(
            scenario_count,
            nonneg=True,
            name="scenario_common_fairness",
        )

    constraints = []

    # Only DER nodes use a shift parameter.
    for node in range(N):
        if node in DER_NODES:
            constraints.extend([
                v_shift_sq[node] >= VSHIFT_MIN_SQ,
                v_shift_sq[node] <= VSHIFT_MAX_SQ,
            ])

        else:
            constraints.append(v_shift_sq[node] == 0.0)

    if formulation == "fairness_stick":
        if fairness_stick_array is None:
            raise ValueError(
                "fairness_stick_array is required for fairness_stick"
            )

        stick_array = np.asarray(
            fairness_stick_array,
            dtype=float,
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

        # Pairwise three-node fairness-stick relationships.
        constraints.extend([
            v_shift_sq[3] <= k_36 * v_shift_sq[6],
            v_shift_sq[3] <= k_3_12 * v_shift_sq[12],
            v_shift_sq[6] <= k_6_12 * v_shift_sq[12],
        ])


    if fairness_enabled:
        constraints.append(alpha <= 1.0)

    for s, row in scenario_df.iterrows():
        slack_voltage = float(row["slack_voltage_pu"])

        constraints.extend([
            v_sq[s, SLACK] == slack_voltage ** 2,
            v_sq[s, :] >= V_MIN ** 2,
            v_sq[s, :] <= V_MAX ** 2,
        ])

        for node in range(N):
            if node in DER_NODES:
                available = float(pv_available[s, node])

                constraints.extend([
                    p_der[s, node] <= available,
                    p_der[s, node]
                    <= available
                    * (v_shift_sq[node] - v_sq[s, node])
                    / (V2_SQ - V1_SQ),
                ])

                if fairness_enabled:
                    constraints.append(
                        p_der[s, node] >= alpha[s] * available
                    )
            else:
                constraints.append(p_der[s, node] == 0.0)

        # Non-existing branches are fixed to zero.
        for i in range(N):
            for j in range(N):
                if (i, j) not in edge_set:
                    constraints.extend([
                        P[s][i, j] == 0.0,
                        Q[s][i, j] == 0.0,
                    ])

        # LinDistFlow equations.
        # q_DER = 0 and there are no thermal constraints.
        for node in range(1, N):
            upstream = parent[node]

            if children[node]:
                downstream_p = cp.sum(
                    cp.hstack([
                        P[s][node, child]
                        for child in children[node]
                    ])
                )
                downstream_q = cp.sum(
                    cp.hstack([
                        Q[s][node, child]
                        for child in children[node]
                    ])
                )
            else:
                downstream_p = 0.0
                downstream_q = 0.0

            constraints.extend([
                P[s][upstream, node]
                == load_p[s, node] - p_der[s, node] + downstream_p,

                Q[s][upstream, node]
                == load_q[s, node] + downstream_q,

                v_sq[s, node]
                == v_sq[s, upstream]
                - 2.0 * (
                    r[upstream, node] * P[s][upstream, node]
                    + x[upstream, node] * Q[s][upstream, node]
                ),
            ])

    total_der_production = cp.sum(p_der[:, DER_NODES])

    # Average utilisation of each DER across all operating scenarios.
    # Because PV availability is fixed at 10 MW in every scenario, these
    # expressions are affine and the pairwise fairness constraints are linear.
    mean_utilisation_expressions = {
        node: cp.sum(p_der[:, node])
        / float(scenario_count * pv_capacity[node])
        for node in DER_NODES
    }

    if formulation == "disparity_constrained":
        if average_utilisation_disparity_limit is None:
            raise ValueError(
                "average_utilisation_disparity_limit is required "
                "for disparity_constrained"
            )
        epsilon = float(average_utilisation_disparity_limit)
        if epsilon < 0.0:
            raise ValueError(
                "average_utilisation_disparity_limit must be non-negative"
            )

        # Enforce |u_bar_i - u_bar_j| <= epsilon for every DER pair.
        for index_i, node_i in enumerate(DER_NODES):
            for node_j in DER_NODES[index_i + 1:]:
                constraints.extend([
                    mean_utilisation_expressions[node_i]
                    - mean_utilisation_expressions[node_j]
                    <= epsilon,
                    mean_utilisation_expressions[node_j]
                    - mean_utilisation_expressions[node_i]
                    <= epsilon,
                ])

    if formulation in {"utilisation", "disparity_constrained", "fairness_stick"}:
        objective = cp.Maximize(total_der_production)

    elif formulation == "fairness_stage_1":
        objective = cp.Maximize(cp.sum(alpha))

    else:
        if fairness_sum_target is None:
            raise ValueError(
                "fairness_sum_target is required for fairness_stage_2"
            )

        constraints.append(
            cp.sum(alpha)
            >= float(fairness_sum_target) - FAIRNESS_OBJECTIVE_TOL
        )
        objective = cp.Maximize(total_der_production)

    problem = cp.Problem(objective, constraints)

    variables = {
        "P": P,
        "Q": Q,
        "p_der": p_der,
        "v_sq": v_sq,
        "v_shift_sq": v_shift_sq,
        "alpha": alpha,
        "load_p": load_p,
        "load_q": load_q,
        "pv_available": pv_available,
        "total_der_production": total_der_production,
        "mean_utilisation_expressions": mean_utilisation_expressions,
        "average_utilisation_disparity_limit": (
            float(average_utilisation_disparity_limit)
            if average_utilisation_disparity_limit is not None
            else np.nan
        ),
        "fairness_stick_array": (
            np.asarray(fairness_stick_array, dtype=float)
            if fairness_stick_array is not None
            else np.full(3, np.nan)
        ),
    }

    return problem, variables


def extract_joint_solution(
    problem: cp.Problem,
    variables: Dict[str, object],
    scenario_count: int,
) -> Dict[str, object]:
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

    v_sq_value = np.asarray(
        variables["v_sq"].value,
        dtype=float,
    )
    p_der_value = np.asarray(
        variables["p_der"].value,
        dtype=float,
    )
    shift_value = np.asarray(
        variables["v_shift_sq"].value,
        dtype=float,
    )

    if variables["alpha"] is None:
        alpha_value = np.full(scenario_count, np.nan)
    else:
        alpha_value = np.asarray(
            variables["alpha"].value,
            dtype=float,
        )

    average_utilisation_by_node = {
        node: float(
            np.mean(
                p_der_value[:, node]
                / np.maximum(
                    variables["pv_available"][:, node],
                    1e-12,
                )
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
        "alpha": alpha_value,
        "load_p": variables["load_p"],
        "load_q": variables["load_q"],
        "pv_available": variables["pv_available"],
        "total_der_production_MW_scenarios": float(
            np.sum(p_der_value[:, DER_NODES])
        ),
        "average_utilisation_by_node": average_utilisation_by_node,
        "achieved_average_utilisation_disparity": (
            achieved_average_disparity
        ),
        "average_utilisation_disparity_limit": variables[
            "average_utilisation_disparity_limit"
        ],
    }


def solve_utilisation_oriented_envelope(
    scenario_df: pd.DataFrame,
) -> Dict[str, object]:
    problem, variables = build_joint_local_envelope_problem(
        scenario_df=scenario_df,
        formulation="utilisation",
    )
    solve_cvxpy_problem(problem)

    return extract_joint_solution(
        problem,
        variables,
        len(scenario_df),
    )


def solve_fairness_oriented_envelope(
    scenario_df: pd.DataFrame,
) -> Dict[str, object]:
    # Stage 1: obtain the maximum aggregate proportional fairness.
    stage_1_problem, stage_1_variables = (
        build_joint_local_envelope_problem(
            scenario_df=scenario_df,
            formulation="fairness_stage_1",
        )
    )
    solve_cvxpy_problem(stage_1_problem)

    if stage_1_problem.status not in ACCEPTED_STATUSES:
        return {
            "status": stage_1_problem.status,
            "objective_value": np.nan,
        }

    fairness_sum_target = float(stage_1_problem.value)

    # Stage 2: retain the stage-1 fairness optimum and maximise utilisation.
    # This resolves unnecessary freedom in p_DER without changing fairness.
    stage_2_problem, stage_2_variables = (
        build_joint_local_envelope_problem(
            scenario_df=scenario_df,
            formulation="fairness_stage_2",
            fairness_sum_target=fairness_sum_target,
        )
    )
    solve_cvxpy_problem(stage_2_problem)

    result = extract_joint_solution(
        stage_2_problem,
        stage_2_variables,
        len(scenario_df),
    )
    result["stage_1_fairness_sum"] = fairness_sum_target
    return result


def solve_disparity_constrained_envelope(
    scenario_df: pd.DataFrame,
    disparity_limit: float = AVERAGE_UTILISATION_DISPARITY_LIMIT,
) -> Dict[str, object]:
    """
    Maximise total DER production while limiting the difference between the
    average utilisations of every pair of DERs.

    The fairness constraint is:
        |mean_utilisation_i - mean_utilisation_j| <= disparity_limit
    for every participating DER pair.
    """
    problem, variables = build_joint_local_envelope_problem(
        scenario_df=scenario_df,
        formulation="disparity_constrained",
        average_utilisation_disparity_limit=disparity_limit,
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

    problem, variables = build_joint_local_envelope_problem(
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


def solve_fairness_stick_envelope(
    scenario_df: pd.DataFrame,
    target_disparity: float = FAIRNESS_STICK_TARGET_DISPARITY,
    number_of_steps: int = FAIRNESS_STICK_COARSE_STEPS,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Perform a multi-stage search over the three pairwise fairness-stick
    coefficients [k_36, k_3_12, k_6_12].

    Stage 1 searches a coarse Cartesian grid covering the complete specified
    coefficient interval. Stage 2 refines several promising coarse regions.
    Stage 3 performs a final high-resolution search around the best feasible
    candidate identified after the first two stages.

    The selected result is the highest-production candidate satisfying the
    target average utilisation disparity. If no candidate satisfies the target,
    the result with minimum disparity is selected, with production used as the
    secondary criterion.
    """
    if target_disparity < 0.0:
        raise ValueError("target_disparity must be non-negative")
    if not 0.0 < FAIRNESS_STICK_MIN <= FAIRNESS_STICK_MAX <= 1.0:
        raise ValueError(
            "Require 0 < FAIRNESS_STICK_MIN <= "
            "FAIRNESS_STICK_MAX <= 1"
        )
    if number_of_steps < 2:
        raise ValueError("number_of_steps must be at least 2")
    if FAIRNESS_STICK_TOP_CENTRES < 1:
        raise ValueError("FAIRNESS_STICK_TOP_CENTRES must be at least 1")
    if FAIRNESS_STICK_FINE_STEP <= 0.0:
        raise ValueError("FAIRNESS_STICK_FINE_STEP must be positive")
    if FAIRNESS_STICK_FINE_RADIUS < 0:
        raise ValueError("FAIRNESS_STICK_FINE_RADIUS must be non-negative")
    if FAIRNESS_STICK_ULTRA_FINE_STEP <= 0.0:
        raise ValueError("FAIRNESS_STICK_ULTRA_FINE_STEP must be positive")
    if FAIRNESS_STICK_ULTRA_FINE_RADIUS < 0:
        raise ValueError(
            "FAIRNESS_STICK_ULTRA_FINE_RADIUS must be non-negative"
        )

    coarse_values = np.linspace(
        FAIRNESS_STICK_MAX,
        FAIRNESS_STICK_MIN,
        number_of_steps,
    )

    sweep_rows: List[Dict[str, object]] = []
    candidate_results: List[Dict[str, object]] = []
    explored = set()

    def evaluate(
        stick_array: Sequence[float],
        search_stage: str,
    ) -> None:
        rounded = tuple(round(float(value), 7) for value in stick_array)
        if rounded in explored:
            return
        explored.add(rounded)

        result = solve_single_fairness_stick(
            scenario_df=scenario_df,
            fairness_stick_array=stick_array,
        )

        if result["status"] in ACCEPTED_STATUSES:
            disparity = float(
                result["achieved_average_utilisation_disparity"]
            )
            production = float(
                result["total_der_production_MW_scenarios"]
            )
            meets_target = disparity <= target_disparity + 1e-8
            v3 = float(result["v_shift_pu"][3])
            v6 = float(result["v_shift_pu"][6])
            v12 = float(result["v_shift_pu"][12])
        else:
            disparity = np.nan
            production = np.nan
            meets_target = False
            v3 = v6 = v12 = np.nan

        k_36, k_3_12, k_6_12 = map(float, stick_array)
        candidate_number = len(sweep_rows)

        sweep_rows.append({
            "candidate": candidate_number,
            "search_stage": search_stage,
            "k_36": k_36,
            "k_3_12": k_3_12,
            "k_6_12": k_6_12,
            "fairness_stick_array": (
                f"[{k_36:.6f}, {k_3_12:.6f}, {k_6_12:.6f}]"
            ),
            "status": result["status"],
            "V_shift_node_3_pu": v3,
            "V_shift_node_6_pu": v6,
            "V_shift_node_12_pu": v12,
            "total_DER_production_MW_scenarios": production,
            "average_utilisation_disparity": disparity,
            "target_disparity": float(target_disparity),
            "meets_target_disparity": meets_target,
        })
        candidate_results.append(result)

    def candidate_ranking_key(index: int) -> Tuple[float, float]:
        """Rank feasible candidates by production, then lower disparity."""
        row = sweep_rows[index]
        return (
            float(row["total_DER_production_MW_scenarios"]),
            -float(row["average_utilisation_disparity"]),
        )

    # Stage 1: complete coarse coverage of the specified three-dimensional
    # coefficient domain.
    for stick_array in product(coarse_values, repeat=3):
        evaluate(stick_array, "coarse")

    valid_coarse_indices = [
        idx
        for idx, row in enumerate(sweep_rows)
        if (
            row["search_stage"] == "coarse"
            and row["status"] in ACCEPTED_STATUSES
        )
    ]

    # Stage 2: refine several promising coarse regions rather than only one.
    if valid_coarse_indices:
        feasible_coarse_indices = sorted(
            [
                idx
                for idx in valid_coarse_indices
                if bool(sweep_rows[idx]["meets_target_disparity"])
            ],
            key=candidate_ranking_key,
            reverse=True,
        )

        low_disparity_indices = sorted(
            valid_coarse_indices,
            key=lambda idx: (
                float(sweep_rows[idx]["average_utilisation_disparity"]),
                -float(
                    sweep_rows[idx][
                        "total_DER_production_MW_scenarios"
                    ]
                ),
            ),
        )

        centre_indices: List[int] = []
        for idx in feasible_coarse_indices + low_disparity_indices:
            if idx not in centre_indices:
                centre_indices.append(idx)
            if len(centre_indices) >= FAIRNESS_STICK_TOP_CENTRES:
                break

        fine_offsets = np.arange(
            -FAIRNESS_STICK_FINE_RADIUS,
            FAIRNESS_STICK_FINE_RADIUS + 1,
        )

        for centre_index in centre_indices:
            centre = np.array([
                sweep_rows[centre_index]["k_36"],
                sweep_rows[centre_index]["k_3_12"],
                sweep_rows[centre_index]["k_6_12"],
            ], dtype=float)

            fine_axes = []
            for value in centre:
                axis = np.clip(
                    value + FAIRNESS_STICK_FINE_STEP * fine_offsets,
                    FAIRNESS_STICK_MIN,
                    FAIRNESS_STICK_MAX,
                )
                fine_axes.append(np.unique(np.round(axis, 7)))

            for stick_array in product(*fine_axes):
                evaluate(stick_array, "fine")

    # Stage 3: centre a final high-resolution grid on the best feasible
    # candidate found in the coarse and fine searches.
    feasible_before_ultra = [
        idx
        for idx, row in enumerate(sweep_rows)
        if (
            row["status"] in ACCEPTED_STATUSES
            and bool(row["meets_target_disparity"])
        )
    ]

    if RUN_ULTRA_FINE_SEARCH and feasible_before_ultra:
        best_index = max(
            feasible_before_ultra,
            key=candidate_ranking_key,
        )
        best_centre = np.array([
            sweep_rows[best_index]["k_36"],
            sweep_rows[best_index]["k_3_12"],
            sweep_rows[best_index]["k_6_12"],
        ], dtype=float)

        ultra_offsets = np.arange(
            -FAIRNESS_STICK_ULTRA_FINE_RADIUS,
            FAIRNESS_STICK_ULTRA_FINE_RADIUS + 1,
        )
        ultra_axes = []
        for value in best_centre:
            axis = np.clip(
                value
                + FAIRNESS_STICK_ULTRA_FINE_STEP * ultra_offsets,
                FAIRNESS_STICK_MIN,
                FAIRNESS_STICK_MAX,
            )
            ultra_axes.append(np.unique(np.round(axis, 7)))

        for stick_array in product(*ultra_axes):
            evaluate(stick_array, "ultra_fine")

    sweep_df = pd.DataFrame(sweep_rows)

    valid_indices = [
        idx
        for idx, result in enumerate(candidate_results)
        if result["status"] in ACCEPTED_STATUSES
    ]
    if not valid_indices:
        return {
            "status": "all_fairness_stick_candidates_failed",
            "objective_value": np.nan,
        }, sweep_df

    target_indices = [
        idx
        for idx in valid_indices
        if bool(sweep_rows[idx]["meets_target_disparity"])
    ]

    if target_indices:
        selected_index = max(
            target_indices,
            key=candidate_ranking_key,
        )
        selection_rule = (
            "highest-production candidate identified by the "
            "multi-stage parameter search among arrays meeting the "
            "target disparity"
        )
    else:
        selected_index = min(
            valid_indices,
            key=lambda idx: (
                float(sweep_rows[idx]["average_utilisation_disparity"]),
                -float(
                    sweep_rows[idx][
                        "total_DER_production_MW_scenarios"
                    ]
                ),
            ),
        )
        selection_rule = (
            "minimum-disparity candidate because no searched array "
            "met the target"
        )

    selected_result = candidate_results[selected_index]
    selected_result["fairness_stick_target_disparity"] = float(
        target_disparity
    )
    selected_result["fairness_stick_selection_rule"] = selection_rule
    selected_result["fairness_stick_selected_index"] = int(selected_index)
    selected_result["fairness_stick_search_method"] = (
        "full-range coarse grid with multi-centre fine refinement and "
        "optional single-centre ultra-fine refinement"
    )

    sweep_df["selected"] = False
    sweep_df.loc[selected_index, "selected"] = True

    return selected_result, sweep_df


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
            "alpha": (
                float(result["alpha"][s])
                if np.isfinite(result["alpha"][s])
                else np.nan
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
                alpha_value = result["alpha"][s]

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
                    "fairness_lower_bound_MW": (
                        alpha_value * available
                        if np.isfinite(alpha_value)
                        else np.nan
                    ),
                    "fairness_margin_MW": (
                        p_value - alpha_value * available
                        if np.isfinite(alpha_value)
                        else np.nan
                    ),
                    "fairness_active": bool(
                        np.isfinite(alpha_value)
                        and abs(
                            p_value - alpha_value * available
                        ) <= FAIRNESS_BINDING_TOL
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
                    "fairness_lower_bound_MW": np.nan,
                    "fairness_margin_MW": np.nan,
                    "fairness_active": False,
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
        "average_utilisation_disparity_limit": (
            result["average_utilisation_disparity_limit"]
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
        "mean_alpha": scenario_results["alpha"].mean(),
        "minimum_alpha": scenario_results["alpha"].min(),
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
            f"node_{node}_fairness_active_fraction": (
                node_data["fairness_active"].mean()
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


def save_comparison_plots(
    scenario_tables: Dict[str, pd.DataFrame],
) -> None:
    """
    Compare all formulations by scenario.

    scenario_tables keys are short labels and values are scenario-result tables.
    """
    comparison = None

    for label, table in scenario_tables.items():
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        selected = table[[
            "scenario",
            "total_DER_production_MW",
            "absolute_utilisation_difference",
        ]].rename(columns={
            "total_DER_production_MW":
                safe_label + "_total_DER_production_MW",
            "absolute_utilisation_difference":
                safe_label + "_scenario_disparity",
        })

        if comparison is None:
            comparison = selected
        else:
            comparison = comparison.merge(
                selected,
                on="scenario",
                how="inner",
            )

    comparison.to_csv(
        OUTPUT_DIR / "all_formulations_scenario_comparison.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, table in scenario_tables.items():
        ax.plot(
            table["scenario"],
            table["total_DER_production_MW"],
            label=label,
        )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total DER production, MW")
    ax.set_title("Total DER Production Comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "total_DER_production_comparison.png",
        dpi=300,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, table in scenario_tables.items():
        ax.plot(
            table["scenario"],
            table["absolute_utilisation_difference"],
            label=label,
        )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Scenario utilisation disparity")
    ax.set_title("DER Utilisation Disparity Comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "utilisation_disparity_comparison.png",
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
        label="Candidate arrays",
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
    ax.set_xlabel("Pairwise fairness-stick candidate")
    ax.set_ylabel("Total DER production across scenarios, MW")
    ax.set_title("Pairwise Fairness-Stick Search: Production")
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
        label="Candidate arrays",
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
    ax.set_xlabel("Pairwise fairness-stick candidate")
    ax.set_ylabel("Average utilisation disparity")
    ax.set_title("Pairwise Fairness-Stick Search: Disparity")
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
    ax.set_xlabel("Pairwise fairness-stick candidate")
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
    scenario_df = create_scenario_grid()
    scenario_df.to_csv(
        OUTPUT_DIR / "operating_scenarios.csv",
        index=False,
    )

    settings = pd.DataFrame([{
        "number_of_scenarios": len(scenario_df),
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
        "fairness_stick_coarse_steps_per_coefficient": (
            FAIRNESS_STICK_COARSE_STEPS
        ),
        "fairness_stick_top_refinement_centres": (
            FAIRNESS_STICK_TOP_CENTRES
        ),
        "fairness_stick_fine_step": FAIRNESS_STICK_FINE_STEP,
        "fairness_stick_fine_radius": FAIRNESS_STICK_FINE_RADIUS,
        "fairness_stick_ultra_fine_step": (
            FAIRNESS_STICK_ULTRA_FINE_STEP
        ),
        "fairness_stick_ultra_fine_radius": (
            FAIRNESS_STICK_ULTRA_FINE_RADIUS
        ),
        "ultra_fine_search_enabled": RUN_ULTRA_FINE_SEARCH,
        "fairness_stick_target_disparity": (
            FAIRNESS_STICK_TARGET_DISPARITY
        ),
        "fairness_stick_constraints": (
            "V3<=k_36*V6; V3<=k_3_12*V12; V6<=k_6_12*V12 "
            "(all in squared-voltage variables)"
        ),
        "SOC_validation_enabled": RUN_SOC_VALIDATION,
    }])
    settings.to_csv(
        OUTPUT_DIR / "study_settings.csv",
        index=False,
    )

    print("Output directory:", OUTPUT_DIR)
    print("Number of scenarios:", len(scenario_df))
    print("DER nodes:", DER_NODES)
    print(
        "DER capacity at each participating node:",
        PV_CAPACITY_PER_DER_MW,
        "MW",
    )
    print("Base loads multiplied by: 8.0")
    print("Original r and x matrices swapped: yes")
    print("Swapped r and x matrices multiplied by: 0.075")
    print("Reactive-power control: disabled")
    print("Thermal constraints: disabled")
    print(
        "V_shift bounds:",
        np.sqrt(VSHIFT_MIN_SQ),
        "to",
        np.sqrt(VSHIFT_MAX_SQ),
        "p.u.",
    )
    print(
        "Fairness-stick coarse grid:",
        FAIRNESS_STICK_MAX,
        "to",
        FAIRNESS_STICK_MIN,
        "using",
        FAIRNESS_STICK_COARSE_STEPS,
        "values per coefficient",
    )
    print(
        "Fine refinement:",
        FAIRNESS_STICK_TOP_CENTRES,
        "centres, step",
        FAIRNESS_STICK_FINE_STEP,
        "and radius",
        FAIRNESS_STICK_FINE_RADIUS,
    )
    print(
        "Ultra-fine refinement enabled:",
        RUN_ULTRA_FINE_SEARCH,
    )
    print(
        "Target average utilisation disparity:",
        FAIRNESS_STICK_TARGET_DISPARITY,
    )
    print(
        "Constraints: V3 <= k_36*V6, V3 <= k_3_12*V12, "
        "V6 <= k_6_12*V12"
    )

    print(
        "\nSolving Fairness-Stick "
        "Local Export Envelope..."
    )
    fairness_stick_result, fairness_stick_sweep = (
        solve_fairness_stick_envelope(
            scenario_df=scenario_df,
            target_disparity=FAIRNESS_STICK_TARGET_DISPARITY,
            number_of_steps=FAIRNESS_STICK_COARSE_STEPS,
        )
    )

    fairness_stick_sweep.to_csv(
        OUTPUT_DIR / "fairness_stick_sweep.csv",
        index=False,
    )

    save_fairness_stick_sweep_plots(
        fairness_stick_sweep,
        OUTPUT_DIR,
    )

    fairness_stick_outputs = run_formulation(
        "Fairness-Stick Local Export Envelope",
        scenario_df,
        fairness_stick_result,
    )

    fairness_stick_outputs["summary"].to_csv(
        OUTPUT_DIR / "fairness_stick_summary.csv",
        index=False,
    )
    fairness_stick_outputs["shift_results"].to_csv(
        OUTPUT_DIR / "fairness_stick_optimised_V_shift.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("Selected fairness stick")
    print("=" * 80)
    selected_row = fairness_stick_sweep[
        fairness_stick_sweep["selected"]
    ]
    print(selected_row.to_string(index=False))
    print("\nFinished.")
    print("Results saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
