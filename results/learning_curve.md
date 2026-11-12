# Learning Curve: is routing predictability bias-limited?

_Scale: pooled, real_wikipedia gold. Accuracy predicting the strictly-best retriever, 5-fold stratified CV x 5 repeats, subsampled stratified._

| informative n | majority | GBT | lift | spread across draws |
|---:|---:|---:|---:|---:|
| 250 | 60.8% | 53.3% | **-7.5%** | +/-2.2 |
| 500 | 60.8% | 54.9% | **-5.9%** | +/-0.7 |
| 1000 | 60.9% | 56.7% | **-4.2%** | +/-0.6 |
| 1500 | 60.9% | 58.0% | **-2.8%** | +/-0.5 |
| 2000 | 60.9% | 58.6% | **-2.3%** | +/-0.3 |
| 3175 | 60.9% | 59.9% | **-1.0%** | +/-0.0 |

Lift moves +6.6 points between the smallest and largest sample. 

A flat or plateauing curve means the barrier is the state representation rather than sample size: collecting more queries would not change the conclusion. A curve still climbing at the largest n would mean the opposite, and would make scaling the query count worth its compute.

Full-set point reproduces `results/learnability_analysis.json`: **yes**.
