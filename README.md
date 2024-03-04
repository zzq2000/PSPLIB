# PSPLIB

启动程序(jobs = 30, Gamma = 5)
```
python main.py --jobs 30 --Gamma 5 > "./logs/jobs_30_Gamma_5.log" 2>&1
```

目前限制之间存在优化冲突，具体的冲突在model.ilp中，大概率是可达性变量z的问题。