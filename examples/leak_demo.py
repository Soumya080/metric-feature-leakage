"""Reproduce reward-into-feature leakage from nothing, in about thirty lines.

No corpus, no retrieval, no downloads. `make_synthetic` builds a routing problem in which the
best arm is chosen INDEPENDENTLY of the features, so the target is unlearnable by
construction: any accuracy above the majority-class rate can only have come from leakage this
script introduces on purpose.

Then each rung of the taxonomy is injected into a single feature column, and the same
classifier is refit. The point the table makes is not that a direct leak helps -- everyone
expects that -- but that a NOISED leak helps almost as much, and that one leaked column out
of twenty is enough. That is why an accidental leak is dangerous: it does not need to be
clean, and it does not need to be widespread.

Run:  python examples/leak_demo.py
"""

import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import leakcheck
from leakcheck import (CORRELATED_PROXY, DIRECT, JITTERED, MONOTONE_DERIVED, SINGLE_ARM,
                       inject, make_synthetic)

X, R = make_synthetic(n=1200, n_arms=3, n_features=20, best_arm_share=0.60, seed=0)
y = R.argmax(axis=1)
majority = np.bincount(y).max() / len(y)


def accuracy(features):
    # Deliberately small: 40 trees on 1,200 rows keeps the whole demo under ~20 seconds, and
    # the effect being shown is so large that model capacity is not what decides it.
    model = GradientBoostingClassifier(n_estimators=40, random_state=0)
    return cross_val_score(model, features, y, cv=5, scoring="accuracy", n_jobs=1).mean()


print(f"\n{len(X)} rows, {X.shape[1]} features, {R.shape[1]} arms")
print(f"The winning arm was drawn independently of the features, so the majority-class rate "
      f"of {majority:.1%}\nis the ceiling for any honest model. Anything above it is leakage.\n")
print(f"{'feature set':22s} {'accuracy':>9s} {'vs majority':>12s}  {'leakcheck verdict':22s}")
print("-" * 74)

rows = [("honest (no leak)", X)] + [
    (rung.replace("_", " "), inject(X, R, rung)) for rung in
    (DIRECT, SINGLE_ARM, MONOTONE_DERIVED, JITTERED, CORRELATED_PROXY)
]

scores = {}
for label, features in rows:
    acc = accuracy(features)
    scores[label] = acc
    report = leakcheck.audit(features, R)
    if report.leaked:
        verdict = f"LEAK {report.worst.rung}"
    elif report.suspicions:
        verdict = f"suspect {report.worst.rung}"
    else:
        verdict = "clean"
    print(f"{label:22s} {acc:8.1%} {acc - majority:+11.1%}  {verdict:22s}")

# DIRECT and JITTERED leak the same arms in the same columns, so the only difference between
# them is noise -- which makes this ratio a clean answer to "what does a leak lose when it
# stops being an exact copy?" Computed rather than quoted, so editing the demo cannot make the
# prose wrong.
gain_direct = scores["direct"] - majority
gain_jitter = scores["jittered"] - majority
retained = gain_jitter / gain_direct if gain_direct > 0 else float("nan")

print(f"""
DIRECT vs JITTERED is the comparison to study. Both leak all {R.shape[1]} arms into the same
{R.shape[1]} columns; the only difference is that JITTERED adds noise. The leak still retains
{retained:.0%} of its advantage ({gain_jitter:+.1%} of {gain_direct:+.1%} over the majority rate).

That is the finding: a leak does not have to be a clean copy of the reward to invert a
conclusion, and a noised copy is what an accident actually produces. An exact copy is easy to
catch with an equality check -- a noised one is invisible to equality checks and still does
most of the damage. How much it retains depends on the form of the perturbation: on the retrieval corpus this
package was extracted from, a value-keyed +/-15% recoding (not noise) kept essentially all of
the advantage (72.7% vs 73.1% accuracy against a 60.9% majority baseline), while independent
additive noise of +/-0.15 kept about half (54%).

SINGLE ARM matters too: leaking one arm out of {R.shape[1]}, one column out of {X.shape[1]},
still buys {scores['single arm'] - majority:+.1%}.

The honest row is the control. It sits at or below the majority rate because the label was
drawn independently of the features -- the only way to be certain that everything below it is
leakage and not a real signal you happened to find.

CORRELATED PROXY is the rung leakcheck deliberately will not call, and it still buys
{scores['correlated proxy'] - majority:+.1%}. It is a noisy observation of the winning arm rather than a function
of the reward, so from two arrays alone it is indistinguishable from a genuinely excellent
difficulty feature. Resolve that one by retraining without the column, never by inspecting it.
""")
