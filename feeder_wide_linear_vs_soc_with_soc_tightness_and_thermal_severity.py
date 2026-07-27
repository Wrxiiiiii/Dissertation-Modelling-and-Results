import os
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_feeder_wide"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DER_NODES = [6, 12]

# Scenario grid for the full 10 x 10 study
SLACK_VOLTAGE_MIN = 0.98
SLACK_VOLTAGE_MAX = 1.03
N_SLACK_STEPS = 10

LOAD_FACTOR_MIN = 0.2
LOAD_FACTOR_MAX = 0.8
N_LOAD_STEPS = 10

# Thermal constraint setting for the LinDistFlow OE optimisation.
# The SOC validation load flow itself deliberately does not impose thermal limits;
# they are checked afterwards using the solved SOC currents.
USE_THERMAL_CONSTRAINT = True

# Reactive-power control settings used in the comparison case.
QMAX_FRAC = 0.25
LOCAL_VOLTAGE_THRESHOLD = 1.04
VOLTAGE_BINDING_TOL = 1e-3
THERMAL_BINDING_TOL = 1e-3
CAPACITY_BINDING_TOL = 1e-4

# SOC-relaxation tightness tolerance.
# 1e-5 = 0.00001. A branch is classified as numerically tight when
# |v_i * L_ij - (P_ij^2 + Q_ij^2)| <= SOC_TIGHTNESS_TOL.
SOC_TIGHTNESS_TOL = 1e-5

# Post-processing tolerance used when deciding whether the SOC load-flow
# current exceeds the line thermal rating. This is not an SOC constraint:
# thermal limits remain deliberately excluded from the SOC validation model.
THERMAL_VIOLATION_TOL = 1e-6

print("Output directory:", OUTPUT_DIR)
print("Number of scenarios:", N_SLACK_STEPS * N_LOAD_STEPS)

N = 13
slack = 0

base_l_P = np.array([
    0, 0.2, 0, 0.4, 0.17, 0.23, 1.155,
    0, 0.17, 0.843, 0, 0.17, 0.128
])

base_l_Q = np.array([
    0, 0.116, 0, 0.29, 0.125, 0.132,
    0.66, 0, 0.151, 0.462, 0, 0.08, 0.086
])

v_min = 0.95
v_max = 1.05

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

edges = [(i, j) for i in range(N) for j in range(N) if A[i, j] == 1]

children = {i: [] for i in range(N)}
parent = {}
for i, j in edges:
    children[i].append(j)
    parent[j] = i

print("Number of feeder branches:", len(edges))

pv_capacity = np.zeros(N)
pv_capacity[12] = 1.5
pv_capacity[6] = 3.5

def get_pv_available():
    """Full PV availability is assumed in this network-stress scenario grid."""
    pv_available = np.zeros(N)
    for j in DER_NODES:
        pv_available[j] = pv_capacity[j]
    return pv_available

def create_scenario_grid():
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

    rows = []
    scenario = 0
    for slack_voltage_pu in slack_values:
        for load_factor in load_values:
            rows.append({
                "scenario": scenario,
                "slack_voltage_pu": float(slack_voltage_pu),
                "load_factor": float(load_factor),
            })
            scenario += 1

    return pd.DataFrame(rows)

# ============================================================
# Feeder-wide OE solver
# ============================================================

def solve_feeder_wide_export_oe(
    l_P,
    l_Q,
    pv_available,
    slack_voltage_pu,
    use_reactive_support=False,
    qmax_frac=0.0,
    verbose=False,
):
    """
    Maximise total accommodated DER production.

    The function returns branch flows as well as DER production and
    voltages so that voltage, thermal, and capacity constraints can
    be diagnosed after the optimisation.
    """
    P = cp.Variable((N, N))
    Q = cp.Variable((N, N))
    v = cp.Variable(N)

    p_der = cp.Variable(N, nonneg=True)
    q_der = cp.Variable(N)

    constraints = [
        v[slack] == slack_voltage_pu ** 2,
        v >= v_min ** 2,
        v <= v_max ** 2,
    ]

    for j in range(N):
        if j in DER_NODES:
            constraints += [p_der[j] <= pv_available[j]]
        else:
            constraints += [
                p_der[j] == 0,
                q_der[j] == 0,
            ]

    # Local voltage-based Q droop.
    if use_reactive_support:
        for j in DER_NODES:
            qmax_j = qmax_frac * pv_capacity[j]
            constraints += [
                q_der[j] == qmax_j * (
                    (v_max ** 2 + v_min ** 2 - 2.0 * v[j])
                    / (v_max ** 2 - v_min ** 2)
                )
            ]
    else:
        constraints += [q_der == 0]

    edge_set = set(edges)

    for i in range(N):
        for j in range(N):
            if (i, j) not in edge_set:
                constraints += [
                    P[i, j] == 0,
                    Q[i, j] == 0,
                ]

    for j in range(1, N):
        i = parent[j]

        downstream_P = cp.sum([P[j, k] for k in children[j]])
        downstream_Q = cp.sum([Q[j, k] for k in children[j]])

        net_P = l_P[j] - p_der[j]
        net_Q = l_Q[j] - q_der[j]

        constraints += [
            P[i, j] == net_P + downstream_P,
            Q[i, j] == net_Q + downstream_Q,
            v[j] == v[i] - 2.0 * (
                r[i, j] * P[i, j]
                + x[i, j] * Q[i, j]
            ),
        ]

        if USE_THERMAL_CONSTRAINT:
            constraints += [
                cp.norm(cp.hstack([P[i, j], Q[i, j]]), 2)
                <= I_max[i, j]
            ]

    oe_total = cp.sum(p_der[DER_NODES])
    problem = cp.Problem(cp.Maximize(oe_total), constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=verbose)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "status": problem.status,
            "OE_total_MW": np.nan,
            "p_der": np.full(N, np.nan),
            "q_der": np.full(N, np.nan),
            "voltage": np.full(N, np.nan),
            "P": np.full((N, N), np.nan),
            "Q": np.full((N, N), np.nan),
        }

    voltage = np.sqrt(np.maximum(v.value, 0.0))

    return {
        "status": problem.status,
        "OE_total_MW": float(np.sum(p_der.value[DER_NODES])),
        "p_der": np.asarray(p_der.value, dtype=float),
        "q_der": np.asarray(q_der.value, dtype=float),
        "voltage": voltage,
        "P": np.asarray(P.value, dtype=float),
        "Q": np.asarray(Q.value, dtype=float),
    }



# ============================================================
# Fixed-injection SOC DistFlow load flow
# ============================================================

ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")


def solve_soc_distflow_loadflow(
    l_P,
    l_Q,
    p_der_fixed,
    q_der_fixed,
    slack_voltage_pu,
    verbose=False,
):
    """
    Solve a fixed-injection SOC branch-flow load flow.

    The active and reactive DER injections are fixed at the values obtained
    from the LinDistFlow OE optimisation. Voltage and thermal operating limits
    are deliberately excluded. The objective minimises resistive line losses.
    """
    n_edges = len(edges)
    edge_index = {edge: idx for idx, edge in enumerate(edges)}

    P_edge = cp.Variable(n_edges, name="soc_P_branch")
    Q_edge = cp.Variable(n_edges, name="soc_Q_branch")
    L_edge = cp.Variable(n_edges, nonneg=True, name="soc_L_current_squared")
    v_sq = cp.Variable(N, name="soc_voltage_squared")

    constraints = [v_sq[slack] == float(slack_voltage_pu) ** 2]

    for j in range(1, N):
        i = parent[j]
        e = edge_index[(i, j)]

        child_p = [P_edge[edge_index[(j, k)]] for k in children[j]]
        child_q = [Q_edge[edge_index[(j, k)]] for k in children[j]]
        downstream_p = cp.sum(cp.hstack(child_p)) if child_p else 0.0
        downstream_q = cp.sum(cp.hstack(child_q)) if child_q else 0.0

        rij = float(r[i, j])
        xij = float(x[i, j])
        net_p = float(l_P[j] - p_der_fixed[j])
        net_q = float(l_Q[j] - q_der_fixed[j])

        constraints += [
            P_edge[e] == net_p + downstream_p + rij * L_edge[e],
            Q_edge[e] == net_q + downstream_q + xij * L_edge[e],
            v_sq[j] == (
                v_sq[i]
                - 2.0 * (rij * P_edge[e] + xij * Q_edge[e])
                + (rij ** 2 + xij ** 2) * L_edge[e]
            ),
            # Rotated second-order cone representation of
            # P_ij^2 + Q_ij^2 <= v_i * L_ij.
            cp.SOC(
                v_sq[i] + L_edge[e],
                cp.hstack([
                    2.0 * P_edge[e],
                    2.0 * Q_edge[e],
                    v_sq[i] - L_edge[e],
                ]),
            ),
        ]

    # A small non-negativity floor is a physical-domain condition, not an
    # operating voltage limit. It prevents square roots of negative values.
    constraints += [v_sq >= 1e-8]

    loss_expression = cp.sum(
        cp.hstack([
            float(r[i, j]) * L_edge[edge_index[(i, j)]]
            for i, j in edges
        ])
    )
    problem = cp.Problem(cp.Minimize(loss_expression), constraints)

    try:
        problem.solve(
            solver=cp.CLARABEL,
            verbose=verbose,
            tol_gap_abs=1e-8,
            tol_gap_rel=1e-8,
            tol_feas=1e-8,
            max_iter=1000,
        )
    except Exception:
        problem.solve(
            solver=cp.SCS,
            verbose=verbose,
            eps=1e-7,
            max_iters=300000,
        )

    nan_matrix = np.full((N, N), np.nan)
    if problem.status not in ACCEPTED_STATUSES or any(
        value is None for value in (
            P_edge.value,
            Q_edge.value,
            L_edge.value,
            v_sq.value,
        )
    ):
        return {
            "status": problem.status,
            "voltage": np.full(N, np.nan),
            "v_squared": np.full(N, np.nan),
            "P": nan_matrix.copy(),
            "Q": nan_matrix.copy(),
            "L": nan_matrix.copy(),
            "current": nan_matrix.copy(),
            "soc_cone_lhs": nan_matrix.copy(),
            "soc_cone_rhs": nan_matrix.copy(),
            "soc_cone_gap": nan_matrix.copy(),
            "soc_cone_abs_gap": nan_matrix.copy(),
            "soc_cone_relative_gap": nan_matrix.copy(),
            "soc_cone_tight": np.zeros((N, N), dtype=bool),
            "soc_max_abs_cone_gap": np.nan,
            "soc_mean_abs_cone_gap": np.nan,
            "soc_max_relative_cone_gap": np.nan,
            "soc_non_tight_branch_count": np.nan,
            "soc_negative_gap_branch_count": np.nan,
            "soc_relaxation_fully_tight": False,
            "losses_MW": np.nan,
        }

    P_value = np.zeros((N, N), dtype=float)
    Q_value = np.zeros((N, N), dtype=float)
    L_value = np.zeros((N, N), dtype=float)
    current_value = np.zeros((N, N), dtype=float)

    v_sq_value = np.asarray(v_sq.value, dtype=float).reshape(-1)

    # Tightness diagnostics for the SOC relaxation.
    # The physical DistFlow equality is
    #     P_ij^2 + Q_ij^2 = v_i * L_ij.
    # The convex SOC model imposes only
    #     P_ij^2 + Q_ij^2 <= v_i * L_ij.
    # Hence gap = RHS - LHS should be zero when the relaxation is tight.
    cone_lhs = np.zeros((N, N), dtype=float)
    cone_rhs = np.zeros((N, N), dtype=float)
    cone_gap = np.zeros((N, N), dtype=float)
    cone_abs_gap = np.zeros((N, N), dtype=float)
    cone_relative_gap = np.zeros((N, N), dtype=float)
    cone_tight = np.zeros((N, N), dtype=bool)

    edge_abs_gaps = []
    edge_relative_gaps = []
    non_tight_count = 0
    negative_gap_count = 0

    for e, (i, j) in enumerate(edges):
        P_value[i, j] = float(P_edge.value[e])
        Q_value[i, j] = float(Q_edge.value[e])
        L_value[i, j] = max(float(L_edge.value[e]), 0.0)
        current_value[i, j] = np.sqrt(L_value[i, j])

        lhs = P_value[i, j] ** 2 + Q_value[i, j] ** 2
        rhs = float(v_sq_value[i]) * L_value[i, j]
        gap = rhs - lhs
        abs_gap = abs(gap)
        # Relative gap is reported only as an additional scale-aware diagnostic.
        # The absolute 1e-5 criterion is used for the tight/non-tight classification.
        scale = max(1.0, abs(lhs), abs(rhs))
        relative_gap = abs_gap / scale

        cone_lhs[i, j] = lhs
        cone_rhs[i, j] = rhs
        cone_gap[i, j] = gap
        cone_abs_gap[i, j] = abs_gap
        cone_relative_gap[i, j] = relative_gap
        cone_tight[i, j] = abs_gap <= SOC_TIGHTNESS_TOL

        edge_abs_gaps.append(abs_gap)
        edge_relative_gaps.append(relative_gap)
        if abs_gap > SOC_TIGHTNESS_TOL:
            non_tight_count += 1
        # A materially negative gap means the numerical solution violates
        # the SOC inequality itself beyond the chosen tolerance.
        if gap < -SOC_TIGHTNESS_TOL:
            negative_gap_count += 1

    voltage = np.sqrt(np.maximum(v_sq_value, 0.0))

    return {
        "status": problem.status,
        "voltage": voltage,
        "v_squared": v_sq_value,
        "P": P_value,
        "Q": Q_value,
        "L": L_value,
        "current": current_value,
        "soc_cone_lhs": cone_lhs,
        "soc_cone_rhs": cone_rhs,
        "soc_cone_gap": cone_gap,
        "soc_cone_abs_gap": cone_abs_gap,
        "soc_cone_relative_gap": cone_relative_gap,
        "soc_cone_tight": cone_tight,
        "soc_max_abs_cone_gap": float(np.max(edge_abs_gaps)),
        "soc_mean_abs_cone_gap": float(np.mean(edge_abs_gaps)),
        "soc_max_relative_cone_gap": float(np.max(edge_relative_gaps)),
        "soc_non_tight_branch_count": int(non_tight_count),
        "soc_negative_gap_branch_count": int(negative_gap_count),
        "soc_relaxation_fully_tight": bool(non_tight_count == 0),
        "losses_MW": float(loss_expression.value),
    }


def calculate_linear_branch_quantities(linear_result):
    """Calculate approximate current and resistive losses from LinDistFlow."""
    current = np.zeros((N, N), dtype=float)
    losses = 0.0
    for i, j in edges:
        sending_voltage = max(float(linear_result["voltage"][i]), 1e-8)
        apparent_flow = float(np.hypot(
            linear_result["P"][i, j],
            linear_result["Q"][i, j],
        ))
        current[i, j] = apparent_flow / sending_voltage
        losses += float(r[i, j]) * current[i, j] ** 2
    return current, float(losses)


def safe_nanmax(values):
    values = np.asarray(values, dtype=float)
    return np.nan if values.size == 0 or np.all(np.isnan(values)) else float(np.nanmax(values))


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    return np.nan if values.size == 0 or np.all(np.isnan(values)) else float(np.nanmean(values))


def safe_rmse(values):
    values = np.asarray(values, dtype=float)
    return np.nan if values.size == 0 or np.all(np.isnan(values)) else float(np.sqrt(np.nanmean(values ** 2)))


# ============================================================
# Run one complete LinDistFlow-versus-SOC validation case
# ============================================================

def run_validation_case(case_name, use_reactive_support, qmax_frac):
    case_output_dir = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    scenario_df = create_scenario_grid()
    pv_available = get_pv_available()

    scenario_rows = []
    node_rows = []
    branch_rows = []
    der_rows = []

    for _, row in scenario_df.iterrows():
        scenario = int(row["scenario"])
        slack_voltage_pu = float(row["slack_voltage_pu"])
        load_factor = float(row["load_factor"])
        l_P_s = base_l_P * load_factor
        l_Q_s = base_l_Q * load_factor

        linear = solve_feeder_wide_export_oe(
            l_P=l_P_s,
            l_Q=l_Q_s,
            pv_available=pv_available,
            slack_voltage_pu=slack_voltage_pu,
            use_reactive_support=use_reactive_support,
            qmax_frac=qmax_frac,
        )

        if linear["status"] in ACCEPTED_STATUSES:
            soc = solve_soc_distflow_loadflow(
                l_P=l_P_s,
                l_Q=l_Q_s,
                p_der_fixed=linear["p_der"],
                q_der_fixed=linear["q_der"],
                slack_voltage_pu=slack_voltage_pu,
            )
        else:
            soc = {
                "status": "not_run",
                "voltage": np.full(N, np.nan),
                "v_squared": np.full(N, np.nan),
                "P": np.full((N, N), np.nan),
                "Q": np.full((N, N), np.nan),
                "L": np.full((N, N), np.nan),
                "current": np.full((N, N), np.nan),
                "soc_cone_lhs": np.full((N, N), np.nan),
                "soc_cone_rhs": np.full((N, N), np.nan),
                "soc_cone_gap": np.full((N, N), np.nan),
                "soc_cone_abs_gap": np.full((N, N), np.nan),
                "soc_cone_relative_gap": np.full((N, N), np.nan),
                "soc_cone_tight": np.zeros((N, N), dtype=bool),
                "soc_max_abs_cone_gap": np.nan,
                "soc_mean_abs_cone_gap": np.nan,
                "soc_max_relative_cone_gap": np.nan,
                "soc_non_tight_branch_count": np.nan,
                "soc_negative_gap_branch_count": np.nan,
                "soc_relaxation_fully_tight": False,
                "losses_MW": np.nan,
            }

        pair_feasible = (
            linear["status"] in ACCEPTED_STATUSES
            and soc["status"] in ACCEPTED_STATUSES
        )

        if pair_feasible:
            linear_current, linear_losses = calculate_linear_branch_quantities(linear)
            voltage_error = soc["voltage"] - linear["voltage"]

            # Branch-level errors across the complete feeder.
            p_errors = np.array([
                soc["P"][i, j] - linear["P"][i, j] for i, j in edges
            ], dtype=float)
            q_errors = np.array([
                soc["Q"][i, j] - linear["Q"][i, j] for i, j in edges
            ], dtype=float)
            current_errors = np.array([
                soc["current"][i, j] - linear_current[i, j] for i, j in edges
            ], dtype=float)

            # The feeder-head active-power difference is dominated by the
            # accumulated feeder losses. To avoid reporting the same physical
            # quantity twice, the internal-branch maximum excludes branches
            # sent directly from the slack bus. Mean errors still use all
            # branches and therefore describe overall agreement.
            internal_edge_indices = np.array([
                idx for idx, (i, _j) in enumerate(edges) if i != slack
            ], dtype=int)

            internal_p_errors = (
                p_errors[internal_edge_indices]
                if internal_edge_indices.size > 0
                else np.array([], dtype=float)
            )
            internal_q_errors = (
                q_errors[internal_edge_indices]
                if internal_edge_indices.size > 0
                else np.array([], dtype=float)
            )
            internal_current_errors = (
                current_errors[internal_edge_indices]
                if internal_edge_indices.size > 0
                else np.array([], dtype=float)
            )

            feeder_head_indices = [
                idx for idx, (i, _j) in enumerate(edges) if i == slack
            ]
            feeder_head_abs_p_error = (
                safe_nanmax(np.abs(p_errors[feeder_head_indices]))
                if feeder_head_indices
                else np.nan
            )

            soc_upper_violation = bool(np.nanmax(soc["voltage"]) > v_max + 1e-6)
            soc_lower_violation = bool(np.nanmin(soc["voltage"]) < v_min - 1e-6)
            # ------------------------------------------------------------
            # Thermal-limit post-processing.
            #
            # The SOC load flow deliberately does NOT impose thermal limits.
            # We therefore check the solved branch currents afterwards and
            # quantify both the frequency and the severity of any exceedance.
            #
            # absolute exceedance = I_SOC - I_max
            # percentage exceedance = (I_SOC - I_max) / I_max * 100
            # loading percentage = I_SOC / I_max * 100
            # ------------------------------------------------------------
            soc_thermal_branches = []
            thermal_exceedance_values = []
            thermal_exceedance_pct_values = []
            thermal_loading_pct_values = []

            for i, j in edges:
                if I_max[i, j] <= 0:
                    continue

                soc_current_ij = float(soc["current"][i, j])
                i_max_ij = float(I_max[i, j])
                loading_pct = 100.0 * soc_current_ij / i_max_ij
                thermal_loading_pct_values.append(loading_pct)

                exceedance = soc_current_ij - i_max_ij
                if exceedance > THERMAL_VIOLATION_TOL:
                    exceedance_pct = 100.0 * exceedance / i_max_ij
                    soc_thermal_branches.append(f"{i}-{j}")
                    thermal_exceedance_values.append(exceedance)
                    thermal_exceedance_pct_values.append(exceedance_pct)

            soc_thermal_violation_branch_count = len(soc_thermal_branches)
            soc_mean_thermal_exceedance = (
                float(np.mean(thermal_exceedance_values))
                if thermal_exceedance_values else 0.0
            )
            soc_max_thermal_exceedance = (
                float(np.max(thermal_exceedance_values))
                if thermal_exceedance_values else 0.0
            )
            soc_mean_thermal_exceedance_pct = (
                float(np.mean(thermal_exceedance_pct_values))
                if thermal_exceedance_pct_values else 0.0
            )
            soc_max_thermal_exceedance_pct = (
                float(np.max(thermal_exceedance_pct_values))
                if thermal_exceedance_pct_values else 0.0
            )
            soc_max_thermal_loading_pct = (
                float(np.max(thermal_loading_pct_values))
                if thermal_loading_pct_values else np.nan
            )

            scenario_rows.append({
                "case": case_name,
                "reactive_support": use_reactive_support,
                "qmax_frac": qmax_frac,
                "scenario": scenario,
                "slack_voltage_pu": slack_voltage_pu,
                "load_factor": load_factor,
                "linear_status": linear["status"],
                "soc_status": soc["status"],
                "OE_total_MW": linear["OE_total_MW"],
                "node_6_P_DER_MW": linear["p_der"][6],
                "node_12_P_DER_MW": linear["p_der"][12],
                "linear_min_voltage_pu": float(np.nanmin(linear["voltage"])),
                "linear_max_voltage_pu": float(np.nanmax(linear["voltage"])),
                "soc_min_voltage_pu": float(np.nanmin(soc["voltage"])),
                "soc_max_voltage_pu": float(np.nanmax(soc["voltage"])),
                "mean_abs_voltage_error_pu": safe_nanmean(np.abs(voltage_error)),
                "max_abs_voltage_error_pu": safe_nanmax(np.abs(voltage_error)),
                "rmse_voltage_error_pu": safe_rmse(voltage_error),
                "mean_abs_P_flow_error_MW": safe_nanmean(np.abs(p_errors)),
                "max_abs_internal_P_flow_error_MW": safe_nanmax(np.abs(internal_p_errors)),
                "feeder_head_abs_P_flow_error_MW": feeder_head_abs_p_error,
                "mean_abs_Q_flow_error_MVAr": safe_nanmean(np.abs(q_errors)),
                "max_abs_internal_Q_flow_error_MVAr": safe_nanmax(np.abs(internal_q_errors)),
                "mean_abs_current_error": safe_nanmean(np.abs(current_errors)),
                "max_abs_internal_current_error": safe_nanmax(np.abs(internal_current_errors)),
                "linear_estimated_losses_MW": linear_losses,
                "soc_losses_MW": soc["losses_MW"],
                "loss_difference_MW": soc["losses_MW"] - linear_losses,
                "soc_upper_voltage_violation": soc_upper_violation,
                "soc_lower_voltage_violation": soc_lower_violation,
                "soc_thermal_violation": bool(soc_thermal_branches),
                "soc_thermal_violation_branches": ",".join(soc_thermal_branches),
                "soc_thermal_violation_branch_count": soc_thermal_violation_branch_count,
                "soc_mean_thermal_exceedance_pu": soc_mean_thermal_exceedance,
                "soc_max_thermal_exceedance_pu": soc_max_thermal_exceedance,
                "soc_mean_thermal_exceedance_pct": soc_mean_thermal_exceedance_pct,
                "soc_max_thermal_exceedance_pct": soc_max_thermal_exceedance_pct,
                "soc_max_thermal_loading_pct": soc_max_thermal_loading_pct,
                "soc_tightness_tolerance": SOC_TIGHTNESS_TOL,
                "soc_max_abs_cone_gap": soc["soc_max_abs_cone_gap"],
                "soc_mean_abs_cone_gap": soc["soc_mean_abs_cone_gap"],
                "soc_max_relative_cone_gap": soc["soc_max_relative_cone_gap"],
                "soc_non_tight_branch_count": soc["soc_non_tight_branch_count"],
                "soc_negative_gap_branch_count": soc["soc_negative_gap_branch_count"],
                "soc_relaxation_fully_tight": soc["soc_relaxation_fully_tight"],
            })

            for node in range(N):
                node_rows.append({
                    "case": case_name,
                    "scenario": scenario,
                    "slack_voltage_pu": slack_voltage_pu,
                    "load_factor": load_factor,
                    "node": node,
                    "linear_voltage_pu": linear["voltage"][node],
                    "soc_voltage_pu": soc["voltage"][node],
                    "voltage_error_pu": voltage_error[node],
                    "abs_voltage_error_pu": abs(voltage_error[node]),
                    "P_DER_MW": linear["p_der"][node],
                    "Q_DER_MVAr": linear["q_der"][node],
                    "load_P_MW": l_P_s[node],
                    "load_Q_MVAr": l_Q_s[node],
                })

            for i, j in edges:
                branch_rows.append({
                    "case": case_name,
                    "scenario": scenario,
                    "slack_voltage_pu": slack_voltage_pu,
                    "load_factor": load_factor,
                    "from_node": i,
                    "to_node": j,
                    "linear_P_MW": linear["P"][i, j],
                    "soc_P_MW": soc["P"][i, j],
                    "P_error_MW": soc["P"][i, j] - linear["P"][i, j],
                    "abs_P_error_MW": abs(soc["P"][i, j] - linear["P"][i, j]),
                    "linear_Q_MVAr": linear["Q"][i, j],
                    "soc_Q_MVAr": soc["Q"][i, j],
                    "Q_error_MVAr": soc["Q"][i, j] - linear["Q"][i, j],
                    "abs_Q_error_MVAr": abs(soc["Q"][i, j] - linear["Q"][i, j]),
                    "linear_current": linear_current[i, j],
                    "soc_current": soc["current"][i, j],
                    "current_error": soc["current"][i, j] - linear_current[i, j],
                    "abs_current_error": abs(soc["current"][i, j] - linear_current[i, j]),
                    "is_feeder_head_branch": bool(i == slack),
                    "is_internal_branch": bool(i != slack),
                    "I_max": I_max[i, j],
                    "soc_thermal_loading_pct": (
                        100.0 * soc["current"][i, j] / I_max[i, j]
                        if I_max[i, j] > 0 else np.nan
                    ),
                    "soc_thermal_margin_pu": (
                        I_max[i, j] - soc["current"][i, j]
                        if I_max[i, j] > 0 else np.nan
                    ),
                    "soc_thermal_exceedance_pu": (
                        max(soc["current"][i, j] - I_max[i, j], 0.0)
                        if I_max[i, j] > 0 else np.nan
                    ),
                    "soc_thermal_exceedance_pct": (
                        100.0 * max(
                            soc["current"][i, j] - I_max[i, j],
                            0.0,
                        ) / I_max[i, j]
                        if I_max[i, j] > 0 else np.nan
                    ),
                    "soc_thermal_violation": bool(
                        I_max[i, j] > 0
                        and soc["current"][i, j]
                        > I_max[i, j] + THERMAL_VIOLATION_TOL
                    ),
                    # SOC relaxation tightness diagnostics.
                    "soc_cone_lhs_P2_plus_Q2": soc["soc_cone_lhs"][i, j],
                    "soc_cone_rhs_vL": soc["soc_cone_rhs"][i, j],
                    "soc_cone_gap_vL_minus_P2Q2": soc["soc_cone_gap"][i, j],
                    "soc_cone_abs_gap": soc["soc_cone_abs_gap"][i, j],
                    "soc_cone_relative_gap": soc["soc_cone_relative_gap"][i, j],
                    "soc_cone_tight_at_1e_5": bool(soc["soc_cone_tight"][i, j]),
                    "soc_cone_inequality_numerically_violated": bool(
                        soc["soc_cone_gap"][i, j] < -SOC_TIGHTNESS_TOL
                    ),
                })

            for j in DER_NODES:
                der_rows.append({
                    "case": case_name,
                    "scenario": scenario,
                    "slack_voltage_pu": slack_voltage_pu,
                    "load_factor": load_factor,
                    "DER_node": j,
                    "PV_available_MW": pv_available[j],
                    "P_DER_MW": linear["p_der"][j],
                    "Q_DER_MVAr": linear["q_der"][j],
                    "linear_local_voltage_pu": linear["voltage"][j],
                    "soc_local_voltage_pu": soc["voltage"][j],
                    "local_voltage_error_pu": voltage_error[j],
                    "local_load_MW": l_P_s[j],
                    "curtailment_MW": pv_available[j] - linear["p_der"][j],
                })
        else:
            scenario_rows.append({
                "case": case_name,
                "reactive_support": use_reactive_support,
                "qmax_frac": qmax_frac,
                "scenario": scenario,
                "slack_voltage_pu": slack_voltage_pu,
                "load_factor": load_factor,
                "linear_status": linear["status"],
                "soc_status": soc["status"],
                "OE_total_MW": linear["OE_total_MW"],
            })

    scenario_results = pd.DataFrame(scenario_rows)
    node_results = pd.DataFrame(node_rows)
    branch_results = pd.DataFrame(branch_rows)
    der_results = pd.DataFrame(der_rows)

    scenario_results.to_csv(os.path.join(case_output_dir, f"{case_name}_scenario_comparison.csv"), index=False)
    node_results.to_csv(os.path.join(case_output_dir, f"{case_name}_node_comparison.csv"), index=False)
    branch_results.to_csv(os.path.join(case_output_dir, f"{case_name}_branch_comparison.csv"), index=False)
    der_results.to_csv(os.path.join(case_output_dir, f"{case_name}_DER_results.csv"), index=False)

    paired = scenario_results[
        scenario_results["linear_status"].isin(ACCEPTED_STATUSES)
        & scenario_results["soc_status"].isin(ACCEPTED_STATUSES)
    ].copy()

    # ============================================================
    # Thermal exceedance severity analysis
    # ============================================================
    if not branch_results.empty and "soc_thermal_violation" in branch_results.columns:
        thermal_violation_details = branch_results[
            branch_results["soc_thermal_violation"].astype(bool)
        ].copy()
    else:
        thermal_violation_details = pd.DataFrame()

    thermal_violation_details.to_csv(
        os.path.join(
            case_output_dir,
            f"{case_name}_thermal_violation_details.csv",
        ),
        index=False,
    )

    if not thermal_violation_details.empty:
        thermal_exceedance_pu = thermal_violation_details[
            "soc_thermal_exceedance_pu"
        ].to_numpy(dtype=float)
        thermal_exceedance_pct = thermal_violation_details[
            "soc_thermal_exceedance_pct"
        ].to_numpy(dtype=float)

        total_thermal_violation_branch_instances = int(
            len(thermal_violation_details)
        )
        mean_thermal_exceedance_pu = float(np.mean(thermal_exceedance_pu))
        median_thermal_exceedance_pu = float(np.median(thermal_exceedance_pu))
        p95_thermal_exceedance_pu = float(
            np.percentile(thermal_exceedance_pu, 95)
        )
        max_thermal_exceedance_pu = float(np.max(thermal_exceedance_pu))

        mean_thermal_exceedance_pct = float(np.mean(thermal_exceedance_pct))
        median_thermal_exceedance_pct = float(
            np.median(thermal_exceedance_pct)
        )
        p95_thermal_exceedance_pct = float(
            np.percentile(thermal_exceedance_pct, 95)
        )
        max_thermal_exceedance_pct = float(np.max(thermal_exceedance_pct))
    else:
        total_thermal_violation_branch_instances = 0
        mean_thermal_exceedance_pu = 0.0
        median_thermal_exceedance_pu = 0.0
        p95_thermal_exceedance_pu = 0.0
        max_thermal_exceedance_pu = 0.0
        mean_thermal_exceedance_pct = 0.0
        median_thermal_exceedance_pct = 0.0
        p95_thermal_exceedance_pct = 0.0
        max_thermal_exceedance_pct = 0.0

    thermal_violation_case_fraction = (
        float(paired["soc_thermal_violation"].astype(bool).mean())
        if len(paired) > 0 else np.nan
    )

    mean_max_scenario_thermal_exceedance_pct = (
        float(paired["soc_max_thermal_exceedance_pct"].mean())
        if len(paired) > 0 else np.nan
    )
    overall_max_scenario_thermal_exceedance_pct = (
        float(paired["soc_max_thermal_exceedance_pct"].max())
        if len(paired) > 0 else np.nan
    )

    # Calculate tightness counts once, using an explicit Boolean conversion.
    if not paired.empty:
        tight_cases = int(
            paired["soc_relaxation_fully_tight"].astype(bool).sum()
        )
        non_tight_cases = int(len(paired) - tight_cases)
        tight_fraction = float(tight_cases / len(paired))
    else:
        tight_cases = 0
        non_tight_cases = 0
        tight_fraction = np.nan

    summary = pd.DataFrame([{
        "case": case_name,
        "reactive_support": use_reactive_support,
        "qmax_frac": qmax_frac,
        "number_of_scenarios": len(scenario_results),
        "linear_feasible_cases": int(scenario_results["linear_status"].isin(ACCEPTED_STATUSES).sum()),
        "soc_loadflow_solved_cases": len(paired),
        "mean_total_OE_MW": paired["OE_total_MW"].mean(),
        "mean_abs_voltage_error_pu": paired["mean_abs_voltage_error_pu"].mean(),
        "mean_max_abs_voltage_error_pu": paired["max_abs_voltage_error_pu"].mean(),
        "overall_max_abs_voltage_error_pu": paired["max_abs_voltage_error_pu"].max(),
        "mean_voltage_RMSE_pu": paired["rmse_voltage_error_pu"].mean(),
        "mean_abs_P_flow_error_MW": paired["mean_abs_P_flow_error_MW"].mean(),
        "mean_max_abs_internal_P_flow_error_MW": paired["max_abs_internal_P_flow_error_MW"].mean(),
        "mean_feeder_head_abs_P_flow_error_MW": paired["feeder_head_abs_P_flow_error_MW"].mean(),
        "mean_abs_Q_flow_error_MVAr": paired["mean_abs_Q_flow_error_MVAr"].mean(),
        "mean_max_abs_internal_Q_flow_error_MVAr": paired["max_abs_internal_Q_flow_error_MVAr"].mean(),
        "mean_abs_current_error": paired["mean_abs_current_error"].mean(),
        "mean_max_abs_internal_current_error": paired["max_abs_internal_current_error"].mean(),
        "mean_linear_estimated_losses_MW": paired["linear_estimated_losses_MW"].mean(),
        "mean_soc_losses_MW": paired["soc_losses_MW"].mean(),
        "mean_feeder_head_loss_identity_error_MW": (
            paired["feeder_head_abs_P_flow_error_MW"]
            .sub(paired["soc_losses_MW"])
            .abs()
            .mean()
        ),
        "soc_upper_voltage_violation_cases": int(paired["soc_upper_voltage_violation"].sum()),
        "soc_lower_voltage_violation_cases": int(paired["soc_lower_voltage_violation"].sum()),
        "soc_thermal_violation_cases": int(
            paired["soc_thermal_violation"].astype(bool).sum()
        ),
        "soc_thermal_violation_case_fraction": thermal_violation_case_fraction,
        "total_thermal_violation_branch_instances": total_thermal_violation_branch_instances,
        "mean_thermal_exceedance_pu": mean_thermal_exceedance_pu,
        "median_thermal_exceedance_pu": median_thermal_exceedance_pu,
        "p95_thermal_exceedance_pu": p95_thermal_exceedance_pu,
        "max_thermal_exceedance_pu": max_thermal_exceedance_pu,
        "mean_thermal_exceedance_pct": mean_thermal_exceedance_pct,
        "median_thermal_exceedance_pct": median_thermal_exceedance_pct,
        "p95_thermal_exceedance_pct": p95_thermal_exceedance_pct,
        "max_thermal_exceedance_pct": max_thermal_exceedance_pct,
        "mean_max_scenario_thermal_exceedance_pct": mean_max_scenario_thermal_exceedance_pct,
        "overall_max_scenario_thermal_exceedance_pct": overall_max_scenario_thermal_exceedance_pct,
        "soc_tightness_tolerance": SOC_TIGHTNESS_TOL,
        "soc_relaxation_tight_cases": tight_cases,
        "soc_relaxation_non_tight_cases": non_tight_cases,
        "soc_relaxation_tight_fraction": tight_fraction,
        "overall_max_abs_soc_cone_gap": paired["soc_max_abs_cone_gap"].max(),
        "mean_scenario_max_abs_soc_cone_gap": paired["soc_max_abs_cone_gap"].mean(),
        "mean_soc_cone_abs_gap": paired["soc_mean_abs_cone_gap"].mean(),
        "overall_max_relative_soc_cone_gap": paired["soc_max_relative_cone_gap"].max(),
        "total_non_tight_soc_branches": int(paired["soc_non_tight_branch_count"].sum()),
        "total_negative_soc_gap_branches": int(paired["soc_negative_gap_branch_count"].sum()),
    }])
    summary.to_csv(os.path.join(case_output_dir, f"{case_name}_summary.csv"), index=False)

    # Save branch-level cases where the SOC relaxation is not tight at the
    # selected absolute tolerance. This file will be empty when all solved
    # scenarios are tight.
    if not branch_results.empty and "soc_cone_tight_at_1e_5" in branch_results.columns:
        non_tight_branches = branch_results[~branch_results["soc_cone_tight_at_1e_5"]].copy()
        non_tight_branches.to_csv(
            os.path.join(case_output_dir, f"{case_name}_soc_non_tight_branches.csv"),
            index=False,
        )

    # Thermal severity plots. These are only created if violations exist.
    if not thermal_violation_details.empty:
        plt.figure(figsize=(9, 6))
        plt.hist(
            thermal_violation_details["soc_thermal_exceedance_pct"],
            bins=20,
            edgecolor="black",
        )
        plt.xlabel("SOC thermal exceedance above rating, %")
        plt.ylabel("Number of violating branch instances")
        plt.title(f"Thermal Exceedance Severity: {case_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                case_output_dir,
                f"{case_name}_thermal_exceedance_histogram.png",
            ),
            dpi=300,
        )
        plt.close()

        worst_by_branch = (
            thermal_violation_details
            .groupby(["from_node", "to_node"], as_index=False)
            ["soc_thermal_exceedance_pct"]
            .max()
            .sort_values("soc_thermal_exceedance_pct", ascending=False)
        )
        worst_by_branch.to_csv(
            os.path.join(
                case_output_dir,
                f"{case_name}_thermal_worst_exceedance_by_branch.csv",
            ),
            index=False,
        )

    if not paired.empty:
        plt.figure(figsize=(9, 6))
        plt.scatter(
            paired["scenario"],
            paired["soc_max_thermal_exceedance_pct"],
            alpha=0.75,
        )
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Scenario")
        plt.ylabel("Maximum thermal exceedance in scenario, %")
        plt.title(f"Maximum Thermal Exceedance by Scenario: {case_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                case_output_dir,
                f"{case_name}_max_thermal_exceedance_by_scenario.png",
            ),
            dpi=300,
        )
        plt.close()

        largest = paired.nlargest(10, "max_abs_voltage_error_pu")
        largest.to_csv(os.path.join(case_output_dir, f"{case_name}_largest_voltage_errors.csv"), index=False)

        plt.figure(figsize=(8, 6))
        plt.scatter(node_results["linear_voltage_pu"], node_results["soc_voltage_pu"], alpha=0.55)
        plot_min = min(node_results["linear_voltage_pu"].min(), node_results["soc_voltage_pu"].min())
        plot_max = max(node_results["linear_voltage_pu"].max(), node_results["soc_voltage_pu"].max())
        plt.plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--", label="Perfect agreement")
        plt.xlabel("LinDistFlow voltage, p.u.")
        plt.ylabel("SOC load-flow voltage, p.u.")
        plt.title(f"LinDistFlow vs SOC Voltage: {case_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(case_output_dir, f"{case_name}_linear_vs_soc_voltage.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.scatter(paired["load_factor"], paired["max_abs_voltage_error_pu"], alpha=0.75)
        plt.xlabel("Load factor")
        plt.ylabel("Maximum absolute voltage error, p.u.")
        plt.title(f"Voltage Error vs Load Factor: {case_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(case_output_dir, f"{case_name}_voltage_error_vs_load.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.scatter(paired["slack_voltage_pu"], paired["max_abs_voltage_error_pu"], alpha=0.75)
        plt.xlabel("Slack voltage, p.u.")
        plt.ylabel("Maximum absolute voltage error, p.u.")
        plt.title(f"Voltage Error vs Slack Voltage: {case_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(case_output_dir, f"{case_name}_voltage_error_vs_slack.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.scatter(paired["scenario"], paired["soc_max_abs_cone_gap"], alpha=0.75)
        plt.axhline(SOC_TIGHTNESS_TOL, linestyle="--", label="Tightness tolerance = 1e-5")
        plt.yscale("log")
        plt.xlabel("Scenario")
        plt.ylabel("Maximum absolute SOC cone gap")
        plt.title(f"SOC Relaxation Tightness by Scenario: {case_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(case_output_dir, f"{case_name}_soc_tightness_by_scenario.png"), dpi=300)
        plt.close()

    print("\n" + "=" * 78)
    print("Case:", case_name)
    print("=" * 78)
    print(summary.to_string(index=False))
    if not paired.empty:
        violation_cases = int(
            paired["soc_thermal_violation"].astype(bool).sum()
        )
        print(
            f"Thermal validation: {violation_cases}/{len(paired)} solved scenarios "
            "contain at least one SOC thermal-limit exceedance."
        )
        print(
            "Thermal exceedance severity across violating branch instances: "
            f"mean={mean_thermal_exceedance_pct:.4f}%, "
            f"median={median_thermal_exceedance_pct:.4f}%, "
            f"95th percentile={p95_thermal_exceedance_pct:.4f}%, "
            f"maximum={max_thermal_exceedance_pct:.4f}%."
        )

        print(
            f"SOC tightness: {tight_cases}/{len(paired)} solved scenarios are fully tight "
            f"at tolerance {SOC_TIGHTNESS_TOL:.0e}."
        )
        print(
            "Overall maximum |v_i L_ij - (P_ij^2 + Q_ij^2)|:",
            f"{paired['soc_max_abs_cone_gap'].max():.6e}",
        )
        negative_count = int(paired["soc_negative_gap_branch_count"].sum())
        print(
            "SOC inequality violations beyond tolerance (negative gaps):",
            negative_count,
        )

    return {
        "scenario_results": scenario_results,
        "node_results": node_results,
        "branch_results": branch_results,
        "der_results": der_results,
        "summary": summary,
        "output_dir": case_output_dir,
    }


def compare_validation_cases(no_q_results, q_results):
    combined = pd.concat([no_q_results["summary"], q_results["summary"]], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "linear_vs_soc_no_Q_vs_Q_summary.csv"), index=False)
    print("\nSummary comparison")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    print("\nRunning normal-scenario LinDistFlow versus SOC validation without Q control...")
    no_q_results = run_validation_case(
        case_name="no_Q_control",
        use_reactive_support=False,
        qmax_frac=0.0,
    )

    print("\nRunning normal-scenario LinDistFlow versus SOC validation with Q control...")
    q_results = run_validation_case(
        case_name="with_Q_control_Qmax25pct",
        use_reactive_support=True,
        qmax_frac=QMAX_FRAC,
    )

    compare_validation_cases(no_q_results, q_results)
    print("\nFinished LinDistFlow-versus-SOC load-flow validation.")
