"""leakcheck — find the evaluation metric hiding in your router's input features.

If you train a router, a model selector, or any contextual bandit from logged evaluation data,
you can accidentally give it a function of its own reward as context. It is a one-line
mistake, it is invisible in how papers describe features ("we use retrieval score statistics"
describes both the clean and the leaked version), and it inverts conclusions: on the corpus
this package was extracted from, four leaked columns out of sixty-one moved a router from
below a constant baseline to apparently working.

Usage:

    import leakcheck
    report = leakcheck.audit(X, R)          # X: (n, d) features, R: (n, k) arm rewards
    print(report)
    if report.leaked:
        raise SystemExit("reward leakage in features")

`audit` needs nothing but the two arrays you already have. There is no model to fit, no
train/test split, and no dependency beyond numpy.

WHAT IT WILL AND WILL NOT TELL YOU. Only EXACT agreement between a feature and a reward is
reported as a verdict, because only exact agreement is a fact about your arrays. Everything
else -- monotone transforms, noised copies, correlated proxies -- is reported as a suspicion
with its evidence, and must be resolved by ablation.

That split is not caution for its own sake; it was forced by data. An earlier version called a
verdict on strong rank correlation. Run against RouterBench, it then flagged features derived
from prompt text alone -- which cannot encode a model's score -- at |rho| = 0.88, because those
features identify which benchmark a query came from and benchmark identity predicts which model
wins. A confound through a shared upstream cause is indistinguishable from a monotone leak by
correlation alone, at any threshold. Each rank-based finding therefore carries an arm
SELECTIVITY figure as supporting evidence: a leak of one arm's reward is specific to that arm,
while a confound correlates with all arms at once.

A clean report is evidence, not proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import decode, detectors, taxonomy
from .detectors import (ARGMAX_LIFT_THRESHOLD, EXACT_MATCH_THRESHOLD,
                        RANK_CORRELATION_SUSPECT, RANK_CORRELATION_THRESHOLD)
from .taxonomy import (CORRELATED_PROXY, DECODABLE, DESCRIPTIONS, DIRECT, JITTERED,
                       MONOTONE_DERIVED, RUNGS, SEVERITY, SINGLE_ARM, inject, make_synthetic)

__version__ = "0.2.1"
__all__ = ["audit", "audit_v2", "Finding", "Report", "inject", "make_synthetic", "RUNGS",
           "DIRECT", "MONOTONE_DERIVED", "JITTERED", "SINGLE_ARM", "CORRELATED_PROXY",
           "DECODABLE"]


@dataclass
class Finding:
    """One (feature, arm) pair that carries reward information."""
    feature: int
    arm: int
    rung: str
    statistic: float
    n_eligible: int
    detail: str
    feature_name: Optional[str] = None
    arm_name: Optional[str] = None

    #: Set when the statistic warrants review rather than a call. Always true for rank-based
    #: and proxy findings; only exact agreement yields a verdict.
    suspect: bool = False

    #: How much more this feature correlates with this arm than with the next-best arm.
    #: Arm-specific association suggests a leak; association spread evenly across arms
    #: suggests a confound. NaN for findings where it does not apply.
    selectivity: float = float("nan")

    #: audit_v2 only. Permutation p-value of the held-out decodability R^2, and the R^2 gained
    #: by the feature WITHIN groups over the groups alone (near zero for a pure group confound).
    #: NaN when not computed.
    p_value: float = float("nan")
    within_group_gain: float = float("nan")

    @property
    def is_verdict(self) -> bool:
        """True when this finding is a fact about the arrays, not a judgement call.

        Only exact agreement is a verdict. Every rank-based finding is created with
        suspect=True, and the proxy rung never is a verdict: a confound through a shared cause, a
        laundered reward and a genuinely excellent difficulty feature are not distinguishable from
        the data alone.
        """
        return self.rung != CORRELATED_PROXY and not self.suspect

    def __str__(self) -> str:
        f = self.feature_name or f"features[{self.feature}]"
        a = self.arm_name or f"arm[{self.arm}]"
        tag = "LEAK" if self.is_verdict else "SUSPECT"
        return f"[{tag}] {self.rung:17s} {f} vs {a}: {self.detail}"


@dataclass
class Report:
    n_rows: int
    n_features: int
    n_arms: int
    findings: List[Finding] = field(default_factory=list)
    checked_proxies: bool = True

    @property
    def leaked(self) -> bool:
        """True if any mechanically-decidable leak was found. Proxies do not set this."""
        return any(f.is_verdict for f in self.findings)

    @property
    def verdicts(self) -> List[Finding]:
        return [f for f in self.findings if f.is_verdict]

    @property
    def suspicions(self) -> List[Finding]:
        return [f for f in self.findings if not f.is_verdict]

    @property
    def worst(self) -> Optional[Finding]:
        """Highest-severity finding: rung order first, then statistic."""
        if not self.findings:
            return None
        return min(self.findings, key=lambda f: (SEVERITY.index(f.rung), -f.statistic))

    @property
    def leaked_features(self) -> List[int]:
        return sorted({f.feature for f in self.verdicts})

    @property
    def suspect_features(self) -> List[int]:
        """Columns flagged but not called. These are what to ablate.

        Separate from `leaked_features` because the action differs: a verdict means remove the
        column and regenerate, a suspicion means retrain without it and compare. Any column
        appearing here should be dropped in at least one ablation arm before the result is
        reported.
        """
        return sorted({f.feature for f in self.suspicions if f.feature >= 0})

    def __str__(self) -> str:
        head = (f"leakcheck {__version__}: {self.n_rows} rows, {self.n_features} features, "
                f"{self.n_arms} arms")
        if not self.findings:
            return (f"{head}\n  CLEAN - no feature encodes any arm's reward.\n"
                    f"  (A clean report is evidence, not proof: a correlated proxy through a "
                    f"shared upstream cause can still evade every detector here.)")
        lines = [head]
        if self.verdicts:
            lines.append(f"  {len(self.verdicts)} leak(s) across "
                         f"{len(self.leaked_features)} feature column(s): "
                         f"{self.leaked_features}")
        for f in sorted(self.findings, key=lambda x: (SEVERITY.index(x.rung), -x.statistic)):
            lines.append("  " + str(f))
        if self.suspicions:
            lines.append(f"  {len(self.suspicions)} SUSPECT finding(s) on column(s) "
                         f"{self.suspect_features}. These are NOT verdicts: a confound through "
                         f"a shared upstream cause produces the same signature as a monotone "
                         f"leak. Check the selectivity figures, then resolve by ablation - "
                         f"retrain without these columns and see if the result survives.")
        return "\n".join(lines)


def audit(X, R, feature_names: Optional[Sequence[str]] = None,
          arm_names: Optional[Sequence[str]] = None, check_proxies: bool = True,
          exact_threshold: float = EXACT_MATCH_THRESHOLD,
          rank_threshold: float = RANK_CORRELATION_THRESHOLD,
          suspect_threshold: float = RANK_CORRELATION_SUSPECT,
          proxy_threshold: float = ARGMAX_LIFT_THRESHOLD) -> Report:
    """Audit a feature matrix against a reward table for reward-into-feature leakage.

    Args:
        X: (n, d) feature matrix -- the router's context, exactly as the model sees it.
        R: (n, k) reward table -- R[i, a] is the reward for choosing arm a on row i.
        feature_names / arm_names: optional labels, used only in the report text.
        check_proxies: run the rung-5 proxy scan. Costs a quantile binning per column.
        *_threshold: override the calibrated defaults. Do this only with measured separation
            on your own data; the defaults come from real corpora and are documented in
            CALIBRATION.md.

    Returns:
        Report. `report.leaked` is True iff a mechanically-decidable leak was found, which
        makes the common case a one-line assertion in a test suite.
    """
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=float)
    if X.ndim != 2 or R.ndim != 2:
        raise ValueError(f"X and R must both be 2-D; got {X.shape} and {R.shape}")
    if len(X) != len(R):
        raise ValueError(f"X has {len(X)} rows but R has {len(R)}")

    n, d = X.shape
    k = R.shape[1]
    report = Report(n_rows=n, n_features=d, n_arms=k, checked_proxies=check_proxies)

    def name(i, names, prefix):
        if names is not None and i < len(names):
            return str(names[i])
        return f"{prefix}[{i}]"

    # Which arms leaked at all, so SINGLE_ARM can be distinguished from a broader leak only
    # after every column has been scanned.
    direct_arms = set()
    per_pair_best = {}

    # Rank correlations for every pair, computed up front so per-feature arm selectivity is
    # available when each finding is written. Selectivity is the only signal that distinguishes
    # "monotone function of arm a's reward" from "confounded with every arm at once", and a
    # finding without it is not interpretable -- see detectors.SELECTIVITY_SUGGESTS_LEAK.
    rho_grid = np.zeros((d, k))
    elig_grid = np.zeros((d, k), dtype=int)
    for j in range(d):
        for a in range(k):
            rho_grid[j, a], elig_grid[j, a] = detectors.rank_agreement(X[:, j], R[:, a])

    def selectivity_of(j, a):
        """rho for this arm minus the best rho over the other arms."""
        if k < 2:
            return float("nan")
        others = np.delete(rho_grid[j], a)
        return float(rho_grid[j, a] - others.max())

    for j in range(d):
        for a in range(k):
            rate, n_agree, n_elig, n_distinct = detectors.exact_match(X[:, j], R[:, a])
            if rate > exact_threshold:
                direct_arms.add(a)
                per_pair_best[(j, a)] = Finding(
                    feature=j, arm=a, rung=DIRECT, statistic=rate, n_eligible=n_elig,
                    detail=(f"equal on {n_agree}/{n_elig} non-zero-reward rows "
                            f"({rate:.1%}, {n_distinct} distinct values)"),
                    feature_name=name(j, feature_names, "features"),
                    arm_name=name(a, arm_names, "arm"))
                continue

            rho, n_elig = rho_grid[j, a], int(elig_grid[j, a])
            if rho > suspect_threshold:
                # NEVER a verdict. High rank agreement with low exact agreement is produced
                # both by a monotone/noised leak and by a plain confound -- on RouterBench,
                # prompt-length features reach rho 0.88 against a model's score purely because
                # they identify which benchmark a query came from. Only exact agreement is a
                # fact about the arrays; this is an inference about causation, so it is
                # surfaced with its evidence and left to a human.
                sel = selectivity_of(j, a)
                strong = rho > rank_threshold
                reads_as = ("arm-specific, which is what a real leak looks like"
                            if sel >= detectors.SELECTIVITY_SUGGESTS_LEAK else
                            "spread across arms, which is what a confound looks like")
                per_pair_best[(j, a)] = Finding(
                    feature=j, arm=a, rung=JITTERED if rate > 0 else MONOTONE_DERIVED,
                    statistic=rho, n_eligible=n_elig, suspect=True, selectivity=sel,
                    detail=(f"Spearman rho = {rho:.3f} ({'strong' if strong else 'moderate'}) "
                            f"on {n_elig} non-zero-reward rows with only {rate:.1%} exact "
                            f"agreement; arm selectivity {sel:+.3f} - {reads_as}"),
                    feature_name=name(j, feature_names, "features"),
                    arm_name=name(a, arm_names, "arm"))

    report.findings.extend(per_pair_best.values())

    # A single leaked arm is its own rung, because it is the commonest accidental form and is
    # materially less damaging than leaking all of them. Relabel rather than add.
    if len(direct_arms) == 1:
        for f in report.findings:
            if f.rung == DIRECT:
                f.rung = SINGLE_ARM
                f.detail += " - only this arm's reward leaked"

    if check_proxies:
        leaked = {f.feature for f in report.findings}
        for j in range(d):
            if j in leaked:
                continue  # already a verdict; a proxy note would be noise
            acc, lift, n_strict = detectors.argmax_recovery(X[:, j], R)
            if lift > proxy_threshold:
                report.findings.append(Finding(
                    feature=j, arm=-1, rung=CORRELATED_PROXY, statistic=lift,
                    n_eligible=n_strict,
                    detail=(f"alone recovers the winning arm at {acc:.1%} vs a "
                            f"{acc - lift:.1%} majority rate (+{lift:.1%}) on {n_strict} "
                            f"rows with a strictly-best arm"),
                    feature_name=name(j, feature_names, "features"),
                    arm_name="argmax"))

    return report


def _bh_reject(pvals, q):
    """Benjamini-Hochberg: boolean mask of rejected hypotheses at false-discovery rate q."""
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, p.size + 1) / p.size)
    k = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    out = np.zeros(p.size, dtype=bool)
    out[order[:k]] = True
    return out


def audit_v2(X, R, feature_names: Optional[Sequence[str]] = None,
             arm_names: Optional[Sequence[str]] = None, groups=None, n_perm: int = 200,
             fdr: float = 0.05, min_r2: float = decode.MIN_R2, seed: int = 0, **audit_kwargs) -> Report:
    """`audit`, plus the opt-in decodability test. v1 findings and verdicts are unchanged.

    For every (feature, arm) pair that is not already a verdict, fits the held-out lookup of
    `decode.decodability`. Pairs with R^2 >= min_r2 get a permutation p-value (n_perm
    shuffles of the reward column); those surviving Benjamini-Hochberg at `fdr` become
    DECODABLE suspicions, or have the p-value attached if v1 already flagged them.

    `groups` (length n, e.g. dataset or benchmark id) adds `within_group_gain` to every
    decodability-tested finding. A finding whose gain is below min_r2 while the groups alone
    explain the reward is labelled as reading like a group confound.

    Like every non-exact finding, a DECODABLE one is a suspicion: genuinely predictive
    routing features are decodable too. It is resolved by ablation, never by threshold.
    """
    report = audit(X, R, feature_names=feature_names, arm_names=arm_names, **audit_kwargs)
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=float)
    d, k = X.shape[1], R.shape[1]
    by_pair = {(f.feature, f.arm): f for f in report.findings if f.arm >= 0}

    def name(i, names, prefix):
        return str(names[i]) if names is not None and i < len(names) else f"{prefix}[{i}]"

    tested = []
    for j in range(d):
        for a in range(k):
            f = by_pair.get((j, a))
            if f is not None and f.is_verdict:
                continue
            dec = decode.decodability(X[:, j], R[:, a], groups=groups, seed=seed)
            if dec["r2"] < min_r2:
                continue
            dec["p_value"] = decode.decodability(X[:, j], R[:, a], n_perm=n_perm, seed=seed)["p_value"]
            tested.append((j, a, dec))

    keep = _bh_reject([t[2]["p_value"] for t in tested], fdr)
    for (j, a, dec), ok in zip(tested, keep):
        if not ok:
            continue
        gain = dec["r2_within_group_gain"]
        confound = (groups is not None and gain < min_r2 and dec["r2_group_only"] >= min_r2)
        note = (f"held-out R^2 = {dec['r2']:.3f} from this column alone (permutation p = "
                f"{dec['p_value']:.3g}, n_perm = {n_perm})")
        if groups is not None:
            note += (f"; within groups it adds {gain:+.3f} over the groups' own "
                     f"{dec['r2_group_only']:.3f}"
                     + (" - reads like a GROUP CONFOUND, not a leak" if confound else ""))
        f = by_pair.get((j, a))
        if f is not None:
            f.p_value, f.within_group_gain = dec["p_value"], gain
            f.detail += "; decodability: " + note
        else:
            report.findings.append(Finding(
                feature=j, arm=a, rung=DECODABLE, statistic=dec["r2"], n_eligible=len(R),
                detail=note, feature_name=name(j, feature_names, "features"),
                arm_name=name(a, arm_names, "arm"), suspect=True,
                p_value=dec["p_value"], within_group_gain=gain))
    return report
