#!/usr/bin/env python
# coding: utf-8

# # Computational Study 2-stage gamma-robust Project Scheduling

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from random import seed
from random import randint
from random import random
import gurobipy as gp
from gurobipy import GRB
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

# init gurobipy env
env = gp.Env(params={"OutputFlag":0})
seed(2020)

# 初始化一个列表来存储目标值和时间戳
objective_values = []
times = []

start_time = time.time()


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
    model = gp.Model("Subproblem", env)

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


def mcCormick(factor, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty):
    # 记录当前算法的开始时间
    start_time = time.time()

    # Create the model
    model = gp.Model("McCormick", env)

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
    
    C = range(len(R[0]))
    z = model.addVars([(i, j) for i in V for j in V if i != j], vtype=GRB.BINARY, name="z")
    f = model.addVars([(i, j, k, c) for i in V for j in V if i != j for k in K for c in C], 
                  vtype=GRB.CONTINUOUS, lb=0.0, name="f")


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
    
    for i, j in A:
        model.addConstr(z[a[0],a[1]] == 1)


    for i in V:
        for j in V:
            if i < j:
                model.addConstr(z[i,j] + z[j,i] <= 1)

    for i in V:
        for j in V:
            for p in V:
                if i != j and j != p and i != p:
                    model.addConstr(z[i,j] >= z[i,p] + z[p,j] - 1)
    
    for i in V:
        for j in V:
            if i != j and i != V[-1] and j != V[0]:
                for k in K:
                    for c in C:
                        model.addConstr(f[i,j,k,c] <= min(R[V.index(i)][c], R[V.index(j)][c]) * z[i,j])

    
    for i in V:
        if i != V[-1]:  # Exclude supersink
            for k in K:
                for c in C:
                    model.addConstr(sum(f[i, j, k, c] for j in V if j != i and j != V[0]) == int(R[V.index(i)][c]))

    for j in V:
        if j != V[0]:  # Exclude supersource
            for k in K:
                for c in C:
                    model.addConstr(sum(f[i, j, k, c] for i in V if i != j and i != V[-1]) == int(R[V.index(j)][c]))
    
    for i in V:
        for j in V:
            if i != j and i != V[-1] and j != V[0]:
                for k in K:
                    for c in C:
                        model.addConstr(f[i, j, k, c] >= 0)
    
    for i in range(len(V)):
        model.addConstr(F[i] >= 0)


    # Optimize model
    model.optimize()


    results_F = []
    if model.status == GRB.OPTIMAL:
        for v in model.getVars():
            if (v.VarName).startswith('F'):
                results_F.append(v.X)

        # calculate the exact robust cost for the produced schedule
        adjusted_costs = subproblem(
            results_F, Gamma, V, A, times, uncertainTimes, costs, early_penalty, late_penalty)

        return (adjusted_costs, model)
    
    elif model.status == GRB.INFEASIBLE:
         # 计算IIS
        print('模型是不可行的，正在计算IIS...')
        model.computeIIS()
        
        # IIS信息可以写入到.ilp文件中以便进一步分析
        model.write("model.ilp")
        
        # 打印出不一致的约束
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"{c.ConstrName}: {model.getRow(c)}")
        raise ValueError("模型是不可行的")