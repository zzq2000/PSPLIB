#!/bin/bash

jobs_list=(30 60 90 120)
Gamma_list=(10 5 3)

# 并行执行所有组合的任务
for jobs in "${jobs_list[@]}"; do
    for Gamma in "${Gamma_list[@]}"; do
        python main.py --jobs "$jobs" --Gamma "$Gamma" > "./logs/jobs_${jobs}_Gamma_${Gamma}.log" 2>&1 &
    done
done

# 等待所有后台任务完成
wait