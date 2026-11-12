"""Is the pooled honest cross-validation std in learnability_analysis.json reproducible?

An external review noticed that two standard deviations in the committed artifacts agree to
fifteen significant figures while describing different feature sets: everything_but_leak (57
features, results/paper_claim_checks.json) and pooled honest (61 features,
results/learnability_analysis.json). Two independent 25-fold runs over different feature sets
cannot do that, so one figure is inherited rather than computed.

This runs both configurations at two seeds. If the agreement were real it would persist at
seed 7; it does not, and the fresh honest run does not reproduce the recorded value either.
See results/KNOWN_ISSUES.md for the verdict and for what must not be plotted from that file.

Usage:
    python experiments/router_experiments/check_std_reproducibility.py
"""

import os,sys,json,io
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
R=os.path.abspath('.')
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'experiments','router_experiments'))
from learnability_analysis import ACTIONS,N_SPLITS,N_REPEATS,load_queries,reward_matrix
from canonical_evaluation import get_dataset_paths
paths=get_dataset_paths('')
X=[];Rw=[]
for ds in ('nq','triviaqa','hotpotqa'):
    _,x,r=reward_matrix(load_queries(paths[ds]),'mrr'); X.append(x);Rw.append(r)
X=np.vstack(X);Rw=np.vstack(Rw)
m=(Rw==Rw.max(axis=1,keepdims=True)).sum(axis=1)==1
Xi,yi=X[m],Rw[m].argmax(axis=1)
dim=Xi.shape[1]
cfg={'honest_all61':list(range(dim)),
     'everything_but_leak':[c for c in range(dim) if c not in (22,23,24,25)]}
for seed in (0,7):
    print(f'--- seed {seed} ---')
    for name,cols in cfg.items():
        cv=RepeatedStratifiedKFold(n_splits=N_SPLITS,n_repeats=N_REPEATS,random_state=seed)
        s=cross_val_score(GradientBoostingClassifier(random_state=seed),Xi[:,cols],yi,cv=cv,scoring='accuracy')
        print(f'  {name:22s} acc={s.mean():.10f}  std={s.std():.18f}')
