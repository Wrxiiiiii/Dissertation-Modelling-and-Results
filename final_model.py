import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Study settings
# ============================================================

OUTPUT_DIR = Path("results_joint_local_export_envelope")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 13
SLACK = 0
DER_NODES = [6, 12]

# Keep the original 10 x 10 scenario grid.
SLACK_VOLTAGE_MIN = 0.98
SLACK_VOLTAGE_MAX = 1.03
N_SLACK_STEPS = int(os.environ.get("OE_N_SLACK_STEPS", "10"))

LOAD_FACTOR_MIN = 0.20
LOAD_FACTOR_MAX = 0.80
N_LOAD_STEPS = int(os.environ.get("OE_N_LOAD_STEPS", "10"))

PV_CAPACITY_PER_DER_MW = 1.5

V_MIN = 0.95
V_MAX = 1.05

# Supervisor-provided squared-voltage Volt-Watt formulation.
V1_SQ = 1.04 ** 2
V2_SQ = V_MAX ** 2
VSHIFT_MIN_SQ = 1.02 ** 2
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

base_l_P = np.array([
    0, 0.2, 0, 0.4, 0.17, 0.23, 1.155,
    0, 0.17, 0.843, 0, 0.17, 0.128
])

base_l_Q = np.array([
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
r = x_original.copy()
x = r_original.copy()

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
) -> Tuple[cp.Problem, Dict[str, object]]:
    """
    Construct one optimisation problem containing every operating scenario.

    formulation:
        utilisation
        fairness_stage_1
        fairness_stage_2
    """
    valid_formulations = {
        "utilisation",
        "fairness_stage_1",
        "fairness_stage_2",
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

    fairness_enabled = formulation.startswith("fairness")
    alpha = None
    if fairness_enabled:
        # One common DER participation coefficient in each scenario.
        # It is common across Nodes 6 and 12, while varying by scenario.
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

    if formulation == "utilisation":
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

    No Q support and no thermal constraints are imposed. Thermal ratings are
    not part of the formal study requested by the supervisor.
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
                abs(der_utilisation[0] - der_utilisation[1])
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

    node_6 = der_results[der_results["node"] == 6]
    node_12 = der_results[der_results["node"] == 12]

    summary = {
        "formulation": formulation_name,
        "status": result["status"],
        "number_of_scenarios": len(scenario_results),
        "V_shift_node_6_pu": result["v_shift_pu"][6],
        "V_shift_node_12_pu": result["v_shift_pu"][12],
        "total_DER_production_MW_scenarios": (
            scenario_results["total_DER_production_MW"].sum()
        ),
        "mean_total_DER_production_MW": (
            scenario_results["total_DER_production_MW"].mean()
        ),
        "mean_node_6_utilisation": (
            node_6["utilisation"].mean()
        ),
        "mean_node_12_utilisation": (
            node_12["utilisation"].mean()
        ),
        "mean_absolute_utilisation_difference": (
            scenario_results[
                "absolute_utilisation_difference"
            ].mean()
        ),
        "total_node_6_curtailment_MW_scenarios": (
            node_6["curtailment_MW"].sum()
        ),
        "total_node_12_curtailment_MW_scenarios": (
            node_12["curtailment_MW"].sum()
        ),
        "mean_alpha": scenario_results["alpha"].mean(),
        "minimum_alpha": scenario_results["alpha"].min(),
        "upper_voltage_active_scenario_count": int(
            scenario_results["upper_voltage_active_any"].sum()
        ),
        "lower_voltage_active_scenario_count": int(
            scenario_results["lower_voltage_active_any"].sum()
        ),
        "node_6_capacity_active_fraction": (
            node_6["capacity_active"].mean()
        ),
        "node_12_capacity_active_fraction": (
            node_12["capacity_active"].mean()
        ),
        "node_6_volt_watt_active_fraction": (
            node_6["volt_watt_active"].mean()
        ),
        "node_12_volt_watt_active_fraction": (
            node_12["volt_watt_active"].mean()
        ),
    }

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
    utilisation_scenarios: pd.DataFrame,
    fairness_scenarios: pd.DataFrame,
) -> None:
    comparison = utilisation_scenarios[[
        "scenario",
        "total_DER_production_MW",
        "absolute_utilisation_difference",
    ]].rename(columns={
        "total_DER_production_MW":
            "utilisation_oriented_total_MW",
        "absolute_utilisation_difference":
            "utilisation_oriented_disparity",
    })

    comparison = comparison.merge(
        fairness_scenarios[[
            "scenario",
            "total_DER_production_MW",
            "absolute_utilisation_difference",
        ]].rename(columns={
            "total_DER_production_MW":
                "fairness_oriented_total_MW",
            "absolute_utilisation_difference":
                "fairness_oriented_disparity",
        }),
        on="scenario",
        how="inner",
    )

    comparison["production_cost_of_fairness_MW"] = (
        comparison["utilisation_oriented_total_MW"]
        - comparison["fairness_oriented_total_MW"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "formulation_scenario_comparison.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        comparison["scenario"],
        comparison["utilisation_oriented_total_MW"],
        label="Utilisation-Oriented",
    )
    ax.plot(
        comparison["scenario"],
        comparison["fairness_oriented_total_MW"],
        label="Fairness-Oriented",
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
    ax.plot(
        comparison["scenario"],
        comparison["utilisation_oriented_disparity"],
        label="Utilisation-Oriented",
    )
    ax.plot(
        comparison["scenario"],
        comparison["fairness_oriented_disparity"],
        label="Fairness-Oriented",
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Absolute utilisation difference")
    ax.set_title("DER Utilisation Disparity Comparison")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "utilisation_disparity_comparison.png",
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
        "r_x_swapped": True,
        "reactive_power_control": False,
        "thermal_limits": False,
        "one_shared_shift_per_DER": True,
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
    print("Original r and x matrices swapped: yes")
    print("Reactive-power control: disabled")
    print("Thermal constraints: disabled")
    print("Shared V_shift across all scenarios: yes")

    print(
        "\nSolving Utilisation-Oriented Local Export Envelope..."
    )
    utilisation_result = solve_utilisation_oriented_envelope(
        scenario_df
    )

    print(
        "\nSolving Fairness-Oriented Local Export Envelope..."
    )
    fairness_result = solve_fairness_oriented_envelope(
        scenario_df
    )

    utilisation_outputs = run_formulation(
        "Utilisation-Oriented Local Export Envelope",
        scenario_df,
        utilisation_result,
    )

    fairness_outputs = run_formulation(
        "Fairness-Oriented Local Export Envelope",
        scenario_df,
        fairness_result,
    )

    combined_summary = pd.concat([
        utilisation_outputs["summary"],
        fairness_outputs["summary"],
    ], ignore_index=True)
    combined_summary.to_csv(
        OUTPUT_DIR / "combined_summary.csv",
        index=False,
    )

    combined_shifts = pd.concat([
        utilisation_outputs["shift_results"],
        fairness_outputs["shift_results"],
    ], ignore_index=True)
    combined_shifts.to_csv(
        OUTPUT_DIR / "combined_optimised_V_shift.csv",
        index=False,
    )

    save_comparison_plots(
        utilisation_outputs["scenario_results"],
        fairness_outputs["scenario_results"],
    )

    print("\n" + "=" * 80)
    print("Combined comparison")
    print("=" * 80)
    print(combined_summary.to_string(index=False))
    print("\nFinished.")
    print("Results saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
