import os
from datetime import datetime
import pandas as pd
from itertools import product
from multiprocessing import Pool
import argparse
from tqdm import tqdm
from parse_PSPLIB_dataset import parse_RCPSP
from McCormick_Heuristic import uncertainties, mcCormick

root_data_path = "./dataset/RCPSP"

def test(jobs, Gamma):
    results = pd.DataFrame(columns=[
        "test set",
        "McCormick 1",
        "McCormick 0.5",
        "McCormick 0.25",
        "Time McCormick 1",
        "Time McCormick_0.5",
        "Time McCormick_0.25"
    ])
    combinations = product(range(1, 49), range(1, 11))
    for j, i in tqdm(combinations, total=(48 * 10), desc=f"jobs_{jobs}_Gamma_{Gamma}"):
        V, A, times, K, R = parse_RCPSP(os.path.join(root_data_path, f"j{jobs}.sm.tgz", f"j{jobs}" + str(j) + "_" + str(i) + ".sm")
                                )
        uncertainTimes, costs, early_penalty, late_penalty = uncertainties(
            V, 0, 10, 5, 20, 5, 10, 5, 10)
        Gamma = 10
        time0 = datetime.now()
        costs1, model1 = mcCormick(
            1, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
        time1 = datetime.now()
        costs05, model1_05 = mcCormick(
            1/2, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
        time1_05 = datetime.now()
        costs025, model1_025 = mcCormick(
            1/4, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
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
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tasks with given jobs and Gamma.")
    parser.add_argument("--jobs", type=int, required=True, help="Number of jobs.")
    parser.add_argument("--Gamma", type=int, required=True, help="Gamma value.")
    args = parser.parse_args()

    jobs = args.jobs
    Gamma = args.Gamma

    results = pd.DataFrame(columns=[
        "test set",
        "McCormick 1",
        "McCormick 0.5",
        "McCormick 0.25",
        "Time McCormick 1",
        "Time McCormick_0.5",
        "Time McCormick_0.25"
    ])
    combinations = product(range(1, 49), range(1, 11))
    for j, i in tqdm(combinations, total=(48 * 10), desc=f"jobs_{jobs}_Gamma_{Gamma}"):
        V, A, times, K, R = parse_RCPSP(os.path.join(root_data_path, f"j{jobs}.sm.tgz", f"j{jobs}" + str(j) + "_" + str(i) + ".sm")
                                )
        uncertainTimes, costs, early_penalty, late_penalty = uncertainties(
            V, 0, 10, 5, 20, 5, 10, 5, 10)
        Gamma = 10
        time0 = datetime.now()
        costs1, model1 = mcCormick(
            1, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
        time1 = datetime.now()
        costs05, model1_05 = mcCormick(
            1/2, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
        time1_05 = datetime.now()
        costs025, model1_025 = mcCormick(
            1/4, Gamma, V, A, times, K, R, uncertainTimes, costs, early_penalty, late_penalty)
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
        break

    results.to_csv(f"{jobs}_jobs_{Gamma}_Gamma.csv",
                    index=False, header=True)

                
