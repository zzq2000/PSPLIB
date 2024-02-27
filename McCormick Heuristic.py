#!/usr/bin/env python
# coding: utf-8

# # Computational Study 2-stage gamma-robust Project Scheduling

import os
from datetime import datetime
import numpy as np
import pandas as pd
from random import seed
from random import randint
from random import random
import gurobipy as gp
from gurobipy import GRB
import itertools

# ### Function to read data from PSPLIB and convert it into our format ->> (V,A) and times as tuple [v,time_v]


def read_file(filename):
    data = open(filename)
    input = data.readlines()
    V = []
    A = []
    times = []
    graph = False
    time = False
    for i in input:
        if ("*********************************************" in i):
            graph = False
            time = False
        if (graph == True and not "--------------" in i):
            row = pd.Series(i).str.split().values[0]
            V.append(row[0])
            for j in range(int(row[2])):
                A.append([row[0], row[3 + j]])
        if (time == True and not "--------------" in i):
            row = pd.Series(i).str.split().values[0]
            times.append(row[2])
        if (i == "jobnr.    #modes  #successors   successors\n"):
            graph = True
        if (i == "jobnr. mode duration  R 1  R 2  R 3  R 4\n"):
            time = True
    data.close()

    # make the times integers
    times = list(map(int, times))

    return (V, A, times)


# ### Function to create the (random) uncertain durations for each vertex -> as tuple [v,uncertainty]


def uncertainties(V, uncertainty_lower, uncertainty_upper, cost_lower, cost_upper, early_penalty_lower,
                  early_penalty_upper, late_penalty_lower, late_penalty_upper):
    uncertainTimes = [0]
    costs = [0]
    early_penalty = [0]
    late_penalty = [0]
    for v in range(len(V) - 2):
        uncertainTimes.append(randint(uncertainty_lower, uncertainty_upper))
        a = round(early_penalty_lower + random() *
                  (-early_penalty_lower + early_penalty_upper), 0)
        b = round(late_penalty_lower + random() *
                  (-late_penalty_lower + late_penalty_upper), 0)
        c = round(cost_lower + random() * (-cost_lower + cost_upper), 0)
        costs.append(int(c))
        early_penalty.append(int(a))
        late_penalty.append(int(b + c))
    uncertainTimes.append(0)
    c = round(cost_lower + random() * (-cost_lower + cost_upper), 0)
    costs.append(c)
    early_penalty.append(0)
    late_penalty.append(c + round(late_penalty_lower + random()
                        * (-late_penalty_lower + late_penalty_upper), 0))
    return (uncertainTimes, costs, early_penalty, late_penalty)


# ### Function for solving the instance with the McCormick heuristic

# auxiliary function to solve the subproblem for fixed F, i.e., to get the robust cost together with the cost for the initial schedule
def subproblem(F, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty):
    n = len(V)
    m = len(A)
    M = sum(late_penalty)

    # create the model
    model = gp.Model("Subproblem")

    # create the variales
    x = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    xi_minus = model.addVars(n, vtype=GRB.BINARY, name="xi_minus")
    xi_plus = model.addVars(n, vtype=GRB.BINARY, name="xi_plus")
    q_minus = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="q_minus")
    q_plus = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="q_plus")

    # set the objective
    model.setObjective(gp.quicksum((F[V.index(a[0])] - F[V.index(a[1])] + times[V.index(a[1])]) * x[A.index(a)] + (
        q_plus[A.index(a)] - q_minus[A.index(a)]) * uncertainTimes[V.index(a[1])] for a in A), GRB.MAXIMIZE)

    # Add the constraints:
    model.addConstrs(-early_penalty[V.index(v)] <= gp.quicksum(x[A.index(a)] for a in A if a[1] == v) - gp.quicksum(
        x[A.index(a)] for a in A if a[0] == v) for v in V[1:n])
    model.addConstrs(late_penalty[V.index(v)] >= gp.quicksum(x[A.index(a)] for a in A if a[1] == v) - gp.quicksum(
        x[A.index(a)] for a in A if a[0] == v) for v in V[1:n])
    model.addConstr(gp.quicksum(
        xi_plus[v] + xi_minus[v] for v in range(n)) <= Gamma)
    for a in A:
        model.addConstr(M * xi_plus[V.index(a[1])] +
                        x[A.index(a)] - M <= q_plus[A.index(a)])
        model.addConstr(M * xi_plus[V.index(a[1])] >= q_plus[A.index(a)])
        model.addConstr(M * xi_minus[V.index(a[1])] +
                        x[A.index(a)] - M <= q_minus[A.index(a)])
        model.addConstr(M * xi_minus[V.index(a[1])] >= q_minus[A.index(a)])
        model.addConstr(q_plus[A.index(a)] <= x[A.index(a)])
        model.addConstr(q_minus[A.index(a)] <= x[A.index(a)])

    # Optimize model
    model.optimize()

    # calculate the costs for the baseline schedule
    base_costs = 0
    for i in range(n):
        base_costs = base_costs+F[i]*costs[i]

    return (model.objVal + base_costs)

# main function for the McCormick heuristic


def mcCormick(factor, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty):
    # Create the model
    model = gp.Model("McCormick")

    # Set the constant M
    M = factor*sum(late_penalty)

    # make the times integers
    times = list(map(int, times))

    # Create the variables
    n = len(V)
    m = len(A)
    theta = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    F = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    mu_plus = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="mu_plus")
    mu_minus = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="mu_minus")
    alpha_plus = model.addVars(
        m, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha_plus")
    alpha_minus = model.addVars(
        m, vtype=GRB.CONTINUOUS, lb=0.0, name="alpha_minus")
    beta_plus = model.addVars(m, vtype=GRB.CONTINUOUS,
                              lb=0.0, name="beta_plus")
    beta_minus = model.addVars(
        m, vtype=GRB.CONTINUOUS, lb=0.0, name="beta_minus")
    lambda_plus = model.addVars(
        m, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda_plus")
    lambda_minus = model.addVars(
        m, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda_minus")
    phi_plus = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="phi_plus")
    phi_minus = model.addVars(n, vtype=GRB.CONTINUOUS,
                              lb=0.0, name="phi_minus")

    # Set the objective
    model.setObjective(
        gp.quicksum(early_penalty[v] * mu_minus[v] + late_penalty[v] * mu_plus[v] for v in range(1, n)) + gp.quicksum(
            costs[v] * F[v] + phi_plus[v] + phi_minus[v] for v in range(n)) + Gamma * theta + M * gp.quicksum(
            lambda_plus[a] + lambda_minus[a] for a in range(m)), GRB.MINIMIZE)

    # Add the constraints:
    model.addConstrs(
        theta - M * gp.quicksum(beta_plus[A.index(a)] - lambda_plus[A.index(a)] for a in A if a[1] == v) + phi_plus[
            V.index(v)] >= 0 for v in V)
    model.addConstrs(
        theta - M * gp.quicksum(beta_minus[A.index(a)] - lambda_minus[A.index(a)] for a in A if a[1] == v) + phi_minus[
            V.index(v)] >= 0 for v in V)
    for a in A:
        if a[0] != V[0]:
            model.addConstr(
                mu_plus[V.index(a[1])] - mu_plus[V.index(a[0])] - mu_minus[V.index(a[1])] + mu_minus[V.index(a[0])] -
                alpha_plus[A.index(a)] - alpha_minus[A.index(a)] + lambda_plus[A.index(a)] + lambda_minus[A.index(a)] >=
                F[V.index(a[0])] - F[V.index(a[1])] + times[V.index(a[1])])
        else:
            model.addConstr(
                mu_plus[V.index(a[1])] - mu_minus[V.index(a[1])] - alpha_plus[A.index(a)] - alpha_minus[A.index(a)] +
                lambda_plus[A.index(a)] + lambda_minus[A.index(a)] >= -F[V.index(a[1])] + times[V.index(a[1])])
        model.addConstr(
            alpha_plus[A.index(a)] + beta_plus[A.index(a)] - lambda_plus[A.index(a)] >= uncertainTimes[V.index(a[1])])
        model.addConstr(alpha_minus[A.index(a)] + beta_minus[A.index(a)] - lambda_minus[A.index(a)] >= -uncertainTimes[
            V.index(a[1])])

    # Optimize model
    model.optimize()

    results_F = []
    for v in model.getVars():
        if (v.VarName).startswith('F'):
            results_F.append(v.X)

    # calculate the exact robust cost for the produced schedule
    adjusted_costs = subproblem(
        results_F, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty)

    return (adjusted_costs, model)


# def read_file(filename):
#     with open(filename, 'r') as file:
#         lines = file.readlines()

#     # Initialize parameters
#     V = []  # List of jobs
#     A = []  # List of precedence relations (directed edges)
#     durations = {}  # Durations of jobs
#     resources_info = []  # List to hold resources info
#     r = {}  # Resource requirements for each job

#     # Parsing flags
#     job_section = False
#     resource_section = False

#     # Parse the file
#     for line in lines:
#         # Strip line of whitespace
#         line = line.strip()

#         # Check for job section
#         if 'jobs (incl. supersource/sink )' in line:
#             job_section = True
#             continue

#         # Check for resource section
#         if 'RESOURCEAVAILABILITIES' in line:
#             job_section = False
#             resource_section = True
#             continue

#         # Parse job section
#         if job_section:
#             parts = line.split()
#             # Assuming each job line has 4 parts: job_number, duration, and two resource requirements
#             if len(parts) == 4 and "jobnr." not in parts[0]:
#                 job_number = int(parts[0])
#                 V.append(job_number)
#                 durations[job_number] = int(parts[1])
#                 r[job_number] = list(map(int, parts[2:]))

#         # Parse resource section
#         if resource_section:
#             parts = line.split()
#             if parts and "R 1  R 2  R 3  R 4" not in line:
#                 resources_info = list(map(int, parts))
#                 break  # Assuming resources info is contained in one line

#     # Create E, K, C, R from parsed data
#     # Create a fully connected graph for demonstration
#     E = [(i, j) for i in V for j in V if i != j]
#     # Number of scenarios, this would need to be defined by your problem
#     K = range(1, 2)
#     C = list(range(len(resources_info)))  # Assuming one type of each resource
#     R = resources_info

#     # Assuming the precedence relations are given in a separate section, you would also need to parse that.
#     # For now, we assume that the PSPLIB file format doesn't provide precedence relations, and thus we create a dummy A.
#     A = [(i, j) for i in V for j in V if i < j]  # Dummy precedence relations

#     return V, A, durations, R, r, E, K, C


# def uncertainties(V, uncertainty_lower, uncertainty_upper, cost_lower, cost_upper, early_penalty_lower,
#                   early_penalty_upper, late_penalty_lower, late_penalty_upper):
#     """
#         Function to create the (random) uncertain durations for each vertex -> as tuple [v,uncertainty]
#     """
#     uncertainTimes = [0]
#     costs = [0]
#     early_penalty = [0]
#     late_penalty = [0]
#     for v in range(len(V) - 2):
#         uncertainTimes.append(randint(uncertainty_lower, uncertainty_upper))
#         a = round(early_penalty_lower + random() *
#                   (-early_penalty_lower + early_penalty_upper), 0)
#         b = round(late_penalty_lower + random() *
#                   (-late_penalty_lower + late_penalty_upper), 0)
#         c = round(cost_lower + random() * (-cost_lower + cost_upper), 0)
#         costs.append(int(c))
#         early_penalty.append(int(a))
#         late_penalty.append(int(b + c))
#     uncertainTimes.append(0)
#     c = round(cost_lower + random() * (-cost_lower + cost_upper), 0)
#     costs.append(c)
#     early_penalty.append(0)
#     late_penalty.append(c + round(late_penalty_lower + random()
#                         * (-late_penalty_lower + late_penalty_upper), 0))
#     return (uncertainTimes, costs, early_penalty, late_penalty)


# def subproblem(F, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty, Z_star, M_big):
#     """
#         function to solve the subproblem for fixed instance with the McCormick heuristic
#     """
#     n = len(V)
#     m = len(A)
#     epsilon_t = [(i, j) for (i, j) in itertools.product(
#         range(n), range(n)) if Z_star[i][j] > 0]

#     # create the model
#     model = gp.Model("Subproblem")

#     # create variables
#     x = model.addVars(epsilon_t, name="x", vtype=GRB.CONTINUOUS)
#     xi_plus = model.addVars(n, vtype=GRB.BINARY, name="xi_plus")
#     xi_minus = model.addVars(n, vtype=GRB.BINARY, name="xi_minus")
#     q_plus = model.addVars(epsilon_t, name="q_plus", vtype=GRB.CONTINUOUS)
#     q_minus = model.addVars(epsilon_t, name="q_minus", vtype=GRB.CONTINUOUS)

#     # set the objective
#     objective = gp.quicksum((F[v]-F[w]+times[w]) * x[(v, w)] + costs[(v, w)]
#                             * q_plus[(v, w)] - costs[(v, w)] * q_minus[(v, w)] for (v, w) in epsilon_t)
#     model.setObjective(objective, GRB.MAXIMIZE)

#     # add constraints
#     for v in range(n):
#         # Assuming the first vertex is the starting node
#         if v != 0:
#             model.addConstr(gp.quicksum(x[(i, v)] for i in range(n) if (i, v) in epsilon_t) -
#                             gp.quicksum(x[(v, j)] for j in range(n) if (v, j) in epsilon_t) == uncertainTimes[v])

#     model.addConstr(gp.quicksum(
#         xi_plus[v] + xi_minus[v] for v in range(n)) <= Gamma)

#     for (v, w) in epsilon_t:
#         model.addConstr(q_plus[(v, w)] <= x[(v, w)])
#         model.addConstr(q_minus[(v, w)] <= x[(v, w)])
#         model.addConstr(x[(v, w)] <= M_big * Z_star[v][w])

#         model.addConstr(q_plus[(v, w)] <= M_big * xi_plus[w])
#         model.addConstr(q_minus[(v, w)] <= M_big * xi_minus[w])
#         model.addConstr(q_plus[(v, w)] >= x[(v, w)] - M_big * (1 - xi_plus[w]))
#         model.addConstr(q_minus[(v, w)] >= x[(v, w)] -
#                         M_big * (1 - xi_minus[w]))

#     # optimize the model
#     model.optimize()

#     # calculate the costs for the baseline schedule
#     base_costs = sum(F[i] * costs[i] for i in range(n))

#     return model.objVal + base_costs


# def mcCormick(factor, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty, E, K, C, R, r):
#     """
#         main function for the McCormick heuristic
#     """
#     # Create the model
#     model = gp.Model("McCormick")

#     # Set the constant M
#     M = factor * sum(late_penalty)

#     # Create the variables
#     F = model.addVars(V, name="F")
#     Omega = model.addVar(name="Omega")
#     theta = model.addVars(K, name="theta")
#     lambda_plus = model.addVars(V, K, name="lambda_plus")
#     lambda_minus = model.addVars(V, K, name="lambda_minus")
#     z = model.addVars(itertools.product(V, V), vtype=GRB.BINARY, name="z")
#     f = model.addVars(itertools.product(V, V, K, C), name="f")

#     # Set the objective function
#     model.setObjective(F[max(V)] + Omega, GRB.MINIMIZE)

#     # Add the constraints

#     for k in K:
#         model.addConstr(Omega >= sum((F[v] - F[w] + times[w]) * z[v, w] for v, w in itertools.product(V, V)) +
#                         Gamma * theta[k] +
#                         sum(lambda_plus[v, k] + lambda_minus[v, k] for v in V))

#     for v in V:
#         for k in K:
#             model.addConstr(theta[k] + lambda_plus[v, k] >=
#                             sum(uncertainTimes[V.index(v)] * z[w, v] for w in V))
#             model.addConstr(theta[k] + lambda_minus[v, k] >= -
#                             sum(uncertainTimes[V.index(v)] * z[w, v] for w in V))

#     for i, j in itertools.product(V, V):
#         if i != j:
#             model.addConstr(z[i, j] + z[j, i] <= 1)

#     for i, j, p in itertools.product(V, V, V):
#         if i != j and j != p and i != p:
#             model.addConstr(z[i, j] >= z[i, p] + z[p, j] - 1)

#     for i, j in itertools.product(V, V):
#         if i != j and i != len(V) - 1 and j != 0:
#             for k in K:
#                 for c in C:
#                     model.addConstr(f[i, j, k, c] <= min(
#                         r[i][c], r[j][c]) * z[i, j])

#     for i in V:
#         if i != len(V) - 1:
#             for k in range(1, K + 1):
#                 for c in C:
#                     model.addConstr(sum(f[i, j, k, c]
#                                     for j in V if j != i and j != 0) == r[i][c])

#     for j in V:
#         if j != 0:
#             for k in range(1, K + 1):
#                 for c in C:
#                     model.addConstr(
#                         sum(f[i, j, k, c] for i in V if i != j and i != len(V) - 1) == r[j][c])

#     # Non-negativity and binary constraints are already defined with variable creation

#     # Capacity constraints at origin and destination
#     for k in range(1, K + 1):
#         for c in C:
#             model.addConstr(r[0][c] == R[c])
#             model.addConstr(r[len(V) - 1][c] == R[c])

#     # Flow conservation constraints
#     for a in E:
#         model.addConstr(z[a] <= M)

#     # Optimize model
#     model.optimize()

#     # Post-processing and calculating the exact robust cost for the produced schedule
#     # Extract the solution for z* and use it in the subproblem function
#     Z_star = {a: z[a].X for a in itertools.product(V, V) if z[a].X > 0}
#     epsilon_t = [(i, j) for i, j in Z_star.keys()]
#     adjusted_costs = subproblem(
#         F, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty, epsilon_t, M)

#     return (adjusted_costs, model)


if __name__ == "__main__":

    # Apply the McCormick heuristic algorithms to the test data

    root_data_path = "./dataset"

    jobs_list = [30, 60, 90, 120]
    Gamma_list = [10, 5, 3]

    seed(2020)

    for jobs in jobs_list:
        for Gamma in Gamma_list:
            print(f"{jobs} jobs, Gamma = {Gamma}")
            results = pd.DataFrame({"test set": []})
            for j in range(1, 49):
                for i in range(1, 11):
                    V, A, times = read_file(os.path.join(root_data_path, f"j{jobs}.sm.tgz", f"j{jobs}" + str(j) + "_" + str(i) + ".sm")
                                            )
                    uncertainTimes, costs, early_penalty, late_penalty = uncertainties(
                        V, 0, 10, 5, 20, 5, 10, 5, 10)
                    Gamma = 10
                    time0 = datetime.now()
                    costs1, model1 = mcCormick(
                        1, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty)
                    time1 = datetime.now()
                    costs05, model1_05 = mcCormick(
                        1/2, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty)
                    time1_05 = datetime.now()
                    costs025, model1_025 = mcCormick(
                        1/4, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty)

                    time1_025 = datetime.now()
                    results.loc[len(results)] = pd.Series({
                        "test set": f"j{jobs}" + str(j) + "_" + str(i),
                        "McCormick 1": costs1,
                        "McCormick 0.5": costs05,
                        "McCormick 0.25": costs025,
                        "Time McCormick 1": int((time1 - time0).total_seconds()),
                        "Time McCormick_0.5": int((time1_05 - time1).total_seconds()),
                        "Time McCormick_0.25": int((time1_025 - time1_05).total_seconds()),
                    })
            results.to_csv(f"{jobs}_jobs_{Gamma}_Gamma.csv",
                           index=False, header=True)
