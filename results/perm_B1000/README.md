# Permutation controls, B=1000 — the canonical version

These supersede an earlier B=20 run, which is not included
 

**The B=20 run reported NQ at p=0.048.** That is exactly the floor a B=20 permutation test can
attain under the (1+k)/(1+B) correction, so the one cell that appeared to clear its null had no
resolution below it. At B=1000 the same cell reads **p=0.066** — the effect was an artifact of
the floor.

Pooling moved the other way: **p=0.003**, well clear of the 0.001 floor. So the honest features
do carry signal above chance, and the reason it was invisible per dataset was power rather than
absence — while still landing 1.0 point *below* the majority baseline.

Anything citing the B=20 numbers is stale. `paper/main.tex` was updated in the same commit
that added this file.
