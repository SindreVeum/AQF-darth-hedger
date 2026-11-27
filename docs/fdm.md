# Craig–Sneyd SABR finite differences

The solver now defaults to direct call and put PDEs for all correlations.
The historical pilot/refinement runners below explicitly retain their original
positive-rho parity convention. The current paper-only direct comparison is
documented in [fdm_paper.md](fdm_paper.md).

## Scope and interfaces

`src/fdm.py` solves beta-one SABR with constant parameters. `src/fdm_experiment.py`
runs a small deterministic comparison with the existing LN labels and N=750 ANN
checkpoints. It does not generate a paper-scale dataset or retrain networks.

```python
from src.fdm import sabr_cs_price, label_row
from src.hagan import make_strikes

T, alpha, nu, rho = 1., .12, .85, .45
K = make_strikes(1., T, alpha, rho, nu)
prices = sabr_cs_price(1., K, T, alpha, rho, nu, x_focus=alpha*T**.5)
# prices['price']: OTM put for K<F, call otherwise; all prices undiscounted.
# prices['call']: direct PDE call for every rho.
# positive_rho_parity=True explicitly requests the legacy positive-rho branch.
```

`label_row([T, alpha, nu, rho, K1, ..., K10], **grid)` returns ten decimal Black
IV estimates at F=1. It is a single-grid convenience function, **not a convergence
certificate**. Failed IV inversions remain NaN. The solver accepts nu=0 for Black
limit checks and -1<rho<1. Spatial counts refer to intervals, so there are nx+1
and ny+1 grid points.

## Price convention

For the model

\[
dF_t=\sigma_t F_t\,dW_t^{(1)},\qquad
d\sigma_t=\nu\sigma_t\,dW_t^{(2)},\qquad
d\langle W^{(1)},W^{(2)}\rangle_t=\rho\,dt,
\]

the default FDM implementation directly evolves both option payoffs. The LN
dataset and the historical FDM pilot/refinement experiments retain a different
positive-rho convention: compute a put and define

\[
C_{\rm parity}(F,K,T)=P(F,K,T)+F-K.
\]

With positive_rho_parity=True, FDM evolves only a put surface in this branch; it does not compute a raw call and
then replace it. LN uses the conditional-LN approximation to the put, whereas FDM
uses the original SABR PDE. Their equal price convention does not make the two
models' numerical prices identical.

Directly evolving a call does not itself certify the raw positive-rho payoff
expectation. Its finite-domain boundary/asymptotic conditions must also select
that valuation convention. No boundary conditions were changed by this switch.

For positive correlation, exact lognormal SABR is a strict local martingale.
Consequently, the raw payoff expectation satisfies
\(C_{\rm raw}-P=E[F_T]-K\), and the parity call adds \(F-E[F_T]\).
The parity construction has the fully collateralized interpretation when P is
the exact SABR put. Applying it to the LN put remains an approximation. See
[Piiroinen et al., Sections 2.2–2.3 and Theorem 3.1](https://arxiv.org/pdf/1805.07403).

## PDE and spatial discretization

Homogeneity allows all strikes to share one PDE surface. Set
\(x=\log(F/K)\), \(y=\log(\sigma/\alpha)\), \(\tau=T-t\), and \(V=K u\).
Then

\[
\partial_\tau u=A_0u+A_1u+A_2u,
\quad A_0=\rho\nu\alpha e^y\partial_{xy},
\quad A_1=\tfrac12\alpha^2e^{2y}(\partial_{xx}-\partial_x),
\quad A_2=\tfrac12\nu^2(\partial_{yy}-\partial_y).
\]

The initial normalized put is \((1-e^x)^+\); the call is \((e^x-1)^+\).
The initial payoff kink is replaced by its exact local cell average. Three-point
second-order derivative formulas are used on each nonuniform axis; the mixed
derivative is their tensor product. A cubic spline evaluates the final surface at
\(x=\log(F/K)\), with y=0 exactly on the grid. This is PDE interpolation in log
moneyness, separate from interpolating an ANN's ten IV outputs.

The grids have the form \(q_i=f\sinh(z_i)\), with uniform z and symmetric endpoints.
For the pilot, x is initially in [-8,8], y in [-6,6], and
\(f_x=\min(0.5,\alpha\sqrt T)\), \(f_y=1\). This X focus is held constant across
the refinements for each case. A fixed focus of 0.5 cannot resolve all short-dated,
low-volatility samples; the low-level solver retains that historical default only
for compatibility with the earlier published-example implementation.

X boundaries retain their initial payoffs. Y boundaries impose \(u_y=0\) through
reflected ghost values; the X operator still acts at these Y boundaries and the
mixed term vanishes there. These are **finite-domain project choices**, not claimed
to be McGhee's undisclosed implementation settings. Their effect must be checked by
domain expansion. Prices are not clipped or normalized to disguise errors.

## Original Craig–Sneyd time steps

With h=T/nt and theta=1/2, advance U by

\[
Y_0=U+h(A_0+A_1+A_2)U,\qquad
(I-\theta hA_j)Y_j=Y_{j-1}-\theta hA_jU,\quad j=1,2,
\]
\[
\widetilde Y_0=Y_0+\tfrac12hA_0(Y_2-U),\qquad
(I-\theta hA_j)\widetilde Y_j=\widetilde Y_{j-1}-\theta hA_jU,\quad j=1,2,
\qquad U_{\rm new}=\widetilde Y_2.
\]

Both correction sweeps subtract the old-time directional terms. Directional
tridiagonal matrices are factored once with SciPy LAPACK and reused. There is no
Rannacher startup or alternative ADI correction in this implementation.

## Pilot and acceptance checks

Run from the repository root using the configured Python:

```powershell
python -m pytest -q
python -m src.fdm_experiment
```

The default pilot includes the 2018 Table 6/7 inputs, four rows selected from the
committed test data (one per correlation sign per maturity window), and three
paper-domain corners. The latter cover one-day/5% initial volatility with each
correlation sign and a two-year/50%/positive-correlation case, all at nu_1M=4.
Selection seed is 20261126. Dataset indices and exact parameters are saved.

For each case the script runs:

| Check | Grids (nx, ny, nt) | Last-change threshold |
|---|---|---|
| Space | (200,100,400), (400,200,400), (800,400,400) | 0.5 IV bp |
| Time | (400,200,200), (400,200,400), (400,200,800) | 0.05 IV bp |
| X domain | Extend to at least [-12,12], preserving existing interior nodes | 0.1 IV bp |
| Y domain | Extend to about [-9.919,9.919], preserving existing interior nodes | 0.1 IV bp |

An accepted label requires finite IVs from every run and all last-change thresholds
to pass for that strike. The thresholds are pilot diagnostics, not rigorous error
bounds. Time and domain checks use the medium grid, so interactions with the finest
grid remain a limitation. The script retains both spatial/time changes for judging
the convergence trend. The FDM thresholds are distinct from LN's 0.1 bp quadrature
agreement criterion and must not be interpreted as equal label accuracy guarantees.

Output files under `results/fdm_pilot/`:

- `details.json`: all grid prices, IVs, runtimes, input rows, statuses, package
  versions and source/checkpoint hashes. `complete` distinguishes progress snapshots
  from a completed experiment.
- `labels.csv`: one row per strike; `fdm_iv` is NaN on failed checks while
  `fdm_iv_estimate` retains the fine-grid value. LN/ANN differences against accepted
  labels are NaN when FDM is not accepted.
- `comparison.png`: differences against the fine-grid estimates, with failed FDM
  checks marked explicitly. The figure is a diagnostic, not an accuracy certificate.

The ANN comparison uses the exact same canonical ten input strikes, with no added
smile spline. The published eleven-strike case is not passed to the ANN. Timings
are measured wall time on this machine: FDM includes assembly, solution and IV
inversion; LN includes its quadrature retry logic; ANN excludes checkpoint loading.
Full validation cost and single fine-grid pricing cost should be distinguished.
Rerunning the pilot recomputes every PDE and overwrites only these experiment files.

## Independent controls

The automated tests compare nu=0 against Black and compare small discrete PDEs
against independently assembled matrix exponentials for each correlation sign.
They also retain the earlier negative-correlation numerical regression and check
positive-correlation parity and input validation.

An additional 2011 positive-correlation example (T=1, alpha=.12, nu=.85, rho=.45)
was checked independently. Space refinement changed IVs by 0.810776 then 0.203014
bp, approximately a factor of four. The final time change was 0.000854 bp; X/Y
expansion changes were 0.0000161 bp and approximately zero. Its fine-grid prices
were within 1.084 observed Monte Carlo standard errors of a separate conditional
path simulation using 524,288 antithetic paths and 512/256 nested steps. This has
Monte Carlo uncertainty and is not a deterministic error bound. The maximum
fine-grid FDM/LN IV difference was 3.488 bp, demonstrating that method differences
remain even after adopting the same parity convention. Details and settings are
saved in `results/fdm_positive_reference.json`.

The printed 2018 FDM column is a reproduction target, not an exact numerical oracle.
The earlier independent PDE/Monte Carlo investigation did not reproduce that column
exactly; the new code does not alter formulas or grids to force agreement with it.

## Observed pilot results (2026-11-26)

All eight cases completed, producing 81 candidate IVs. **55 passed the stated
numerical checks; 26 were rejected.** No failed label was filled or promoted to a
training target. The table reports changes across grids, not errors against an
exact SABR oracle.

| Case | Passed | Max space change (IV bp) | Max time change (IV bp) | Max X-domain change (IV bp) |
|---|---:|---:|---:|---:|
| Published 2018 example | 10/11 | 0.653 | 0.00515 | 0.000209 |
| 1D–1Y test row 3571, negative rho | 8/10 | 40.992 | 1.11026 | 0.000082 |
| 1D–1Y test row 44913, positive rho | 7/10 | 17.758 | 0.00646 | <0.000001 |
| 1Y–2Y test row 10745, negative rho | 9/10 | 0.555 | 0.00103 | 0.000007 |
| 1Y–2Y test row 1100, positive rho | 7/10 | 4.117 | 0.02039 | <0.000001 |
| One day, alpha=.05, nu_1M=4, rho=-.9 | 7/10 | 38.234 | 1.07604 | <0.000001 |
| One day, alpha=.05, nu_1M=4, rho=+.9 | 7/10 | 33.298 | 0.83376 | <0.000001 |
| Two years, alpha=.5, nu_1M=4, rho=+.9 | 0/10 | 5.608 | 0.03088 | 119.701 |

Four labels have an invalid IV in at least one grid run. Such labels remain rejected
even when the finest run happens to yield a finite estimate. Most other rejected
labels are wing points with excessive spatial changes. The final corner is strongly
dependent on the X domain and is not usable with the default width of eight.

On the 55 accepted points, LN versus FDM has MAE 1.108 IV bp, 95th-percentile absolute
difference 5.070 bp and maximum 7.162 bp. This deliberately small mixture of published,
random and extreme cases is not a population accuracy estimate.

The canonical ten-strike subset contains 45 accepted points. Comparing all three
methods on exactly those points gives:

| Method | Mean absolute IV difference from accepted FDM (bp) | Maximum (bp) |
|---|---:|---:|
| LN integration | 0.706 | 4.489 |
| Raw-input N=750 ANN | 11.905 | 96.500 |
| Transformed-input N=750 ANN | 2.911 | 16.732 |

These ANN discrepancies combine approximation of the LN labels with LN/FDM method
differences. They must not be confused with the full held-out ANN test error, or
with smile-interpolation error: no ANN spline was used here.

Measured per-case CPU wall times were 62.84–202.53 seconds for a single fine-grid
FDM price/IV calculation, and 112.06–474.19 seconds for all seven FDM checks. LN
label generation took 0.109–0.223 seconds; warm ANN inference took 0.47–1.44 ms,
excluding model loading. Cases 0–1 were completed sequentially, and cases 2–7 in
two additional processes using identical functions and verified source hashes.
Some timings therefore include resource contention; they are not a controlled
throughput benchmark or a basis for extrapolating full-dataset runtime.

An additional X-domain investigation for the final corner is recorded separately
in `results/fdm_domain_check.json`. Keeping the medium-grid internal spacing fixed,
expanding the half-width from 12.1338 to 24.2801 changed IVs by at most 2.39716 bp;
24.2801 to 48.5736 changed them by about 0.0000126 bp. This isolates and substantially
reduces domain sensitivity; a further expansion to 97.1681 changed IVs by only
1.36e-9 bp. It does **not** validate the medium-grid spatial error,
so these diagnostic prices do not replace any rejected pilot labels.

The implementation is a tested experimental FDM reference with explicit acceptance
checks. The present pilot grid is not a validated full-parameter-domain label
generator. Further work should improve wing resolution and choose the X domain
adaptively before generating a complete FDM training dataset.

## Targeted grid and boundary refinement

`python -m src.fdm_refinement` repeats the **same eight saved input rows and 81
strikes**, writing to `results/fdm_refined/`. The original `results/fdm_pilot/`
is retained. `--case NAME --output PATH` can select one of these cases. This is a
small reproduction experiment with explicitly tuned settings; it is not a
general-purpose adaptive grid rule or a full-dataset generator. LN and ANN
predictions are reused from the unchanged baseline, so only FDM is recomputed.

The space/time/X-domain/Y-domain thresholds remain 0.5/0.05/0.1/0.1 IV bp.
Every one of the seven runs must have a finite IV at a strike. A negative coarse
price therefore remains a failed check even if the finer price is positive.
Prices are not clipped, failed IVs are not filled, and Richardson extrapolation
is not used to turn a failed raw-grid estimate into an accepted label.

### Forward-preserving spatial discretization

The standard log-coordinate difference formula is consistent with the SABR PDE,
but generally does not satisfy the discrete identity
\((D_{xx}-D_x)e^x=0\). The resulting finite-grid intrinsic-value error is magnified
when a small OTM call is recovered from a large ITM put by parity. This is a
spatial truncation error, not a different SABR model or a wrong continuous PDE.

The optional `fit_forward=True` replaces the X operator by

\[
A_{1,h}=\tfrac12\alpha^2 e^{2y}(D_{xx}-c_iD_x),\qquad
c_i=\frac{(D_{xx}e^x)_i}{(D_xe^x)_i}.
\]

For neighbor distances \(h_l,h_r\), the implementation evaluates the ratio with
local increments \(\operatorname{expm1}(-h_l)\) and
\(\operatorname{expm1}(h_r)\), avoiding subtraction of nearly equal exponentials.
On a uniform grid, \(c=2\tanh(h/2)/h=1-h^2/12+O(h^4)\). It preserves the
second-order consistency of the smooth-grid discretization while making the X
generator annihilate the forward to floating-point accuracy. The original
Craig–Sneyd stages, conditional LN implementation, and positive-rho parity
convention are unchanged. The legacy default is `False`; the refined pilot
enables it for positive rho and retains the direct-call/direct-put negative-rho
branch. Mixed derivatives still prevent a general positivity guarantee.
This local discrete identity does **not** assert that positive-rho SABR has
\(E[F_T]=F\); the strict-local-martingale distinction and the explicit parity
pricing choice described above remain in force.

An exact Black-limit diagnostic at the low-nu case's strikes isolates this
effect: at nx=400, the old parity implementation has a maximum error of about
29.77 IV bp, compared with about 2.04 bp for direct pricing. Forward fitting
reduces the parity error to about 2.04 bp. Remaining Black error is ordinary
finite-grid error and must still decrease under refinement.

### What the failed points revealed

- The four baseline `invalid_iv` labels originate in **negative coarse-grid
  option prices**, all in the one-day, extreme-correlation cases. The Black
  inversion rejects these correctly. Both X and Y spatial errors matter.
- Merely broadening the sinh focus does not resolve the short-dated wings.
  Reducing the X domain allocates more nodes to relevant moneyness, but must be
  checked independently: a half-width of 0.125 failed the 0.1 bp domain test
  (about 0.314 bp at one positive-rho strike). The refined candidate uses 0.5
  and compares it with a preserved-spacing expansion to at least 1.
- The long/high-vol/positive-rho case needs a much wider X domain. The verified
  half-width 24.280127 is used, with an expansion to at least 48. Space refinement
  is then performed on that domain; domain agreement alone is insufficient.
- The low-nu examples benefit from a smaller X domain, a Y focus scaled to
  \(\nu\sqrt T\), and explicit forward preservation. Their Y domain is still
  expanded separately; a narrower focus is not itself a boundary argument.

The refined pilot keeps spatial doubling, time doubling, and domain extension
separate. Domain extensions retain the old interior sinh nodes:

\[
\Delta z=\frac{2\operatorname{asinh}(W/f)}{n},\qquad
n'=2\left\lceil\frac{\operatorname{asinh}(W_{\rm target}/f)}{\Delta z}\right\rceil,
\qquad W'=f\sinh(n'\Delta z/2).
\]

This avoids confusing an enlarged domain with a coarser interior mesh. Time and
domain checks remain on the medium spatial grid, matching the original pilot
protocol. They are empirical diagnostics, not rigorous total-error bounds.

`comparison.json` records recovered, remaining, and regressed strike indices
(one-based) for each case, plus the old/new fine-solve and validation wall times.
`details.json` retains all seven price/IV arrays, settings, source fingerprints,
and CPU time where measured. Both experiments used concurrent processes for
some cases; their wall-time ratios are **not controlled speedup measurements**.
The full seven-check cost must also be distinguished from one fine-grid solve.

The final candidate medium grids are listed below; spatial coarse/fine grids
halve/double **both** interval counts without changing nt.

| Case | Medium (nx, ny, nt) | X half-width | X focus | Y focus |
|---|---|---:|---:|---:|
| Published example | (400,200,400) | 8 | 0.5 | 1 |
| 1D–1Y negative-rho test row 3571 | (480,240,800) | 8 | alpha sqrt(T) | 1 |
| 1D–1Y positive-rho test row 44913 | (600,160,400) | 1 | alpha sqrt(T) | nu sqrt(T) |
| 1Y–2Y negative-rho test row 10745 | (440,220,400) | 8 | 0.5 | 1 |
| 1Y–2Y positive-rho test row 1100 | (960,256,400) | 1 | alpha sqrt(T) | nu sqrt(T) |
| Both one-day corners | (400,200,800) | 0.5 | alpha sqrt(T) | 1 |
| Two-year high-vol positive-rho corner | (1152,480,400) | 24.280127 | 0.5 | 1 |

All retain Y half-width 6 and the same endpoint boundary formulas. A rejected
trial broadened row 3571's X focus to 0.5 and made its previously passing eighth
strike fail. The final profile restores its original X focus and increases both
spatial resolutions; rejected-trial evidence is retained in `diagnostics.json`.

`remaining_points.csv` lists each unresolved strike's coarse/base/fine prices,
price differences, IV differences and successive price-change ratio. The column
`local_price_change_for_half_iv_bp` is the **local sensitivity scale**

\[
\mathrm{Vega}_{\mathrm{Black}}=\sqrt T\,\phi(d_1),\qquad
\Delta P_{0.5\mathrm{bp}}\approx\mathrm{Vega}_{\mathrm{Black}}\,(5\times10^{-5}),
\]

for F=1 and undiscounted prices. This is not a rigorous error bound or a replacement
for actual IV inversion. Very small wing prices amplify discretization errors;
the observed unresolved prices are not at a double-precision rounding floor.

To measure serial paired costs without repeating all convergence calculations:

```sh
python -m src.fdm_refinement --timing-only
```

Run this after other FDM jobs finish. `timing.json` and `timing.csv` contain two
repetitions per case/configuration, with alternating order and an untimed warm-up.
This compares the same eight cases at **each configuration's coarse grid level**,
including assembly, pricing and IV inversion. The grids and time-step counts
differ by design. These are descriptive cost measurements, not equal-accuracy
speed comparisons, finest-grid runtime predictions, or full-dataset estimates.

## Observed refinement results (2026-11-26)

All eight original cases completed the seven-run protocol. **73 of 81 labels
passed, up from 55; 18 of the 26 previous failures were recovered, with no
previously passing label regressing. Eight remain rejected.** No input rows or
strikes were dropped, no thresholds were relaxed, and no full samples were
generated. The following numbers are grid-to-grid IV changes, not errors against
an exact SABR oracle.

| Case | Passed before → after | Max space change before → after (bp) | Max time change before → after (bp) |
|---|---:|---:|---:|
| Published example | 10/11 → 11/11 | 0.653 → 0.387 | 0.00515 → 0.00514 |
| 1D–1Y negative row 3571 | 8/10 → 8/10 | 40.992 → 27.282 | 1.11026 → 0.25581 |
| 1D–1Y positive row 44913 | 7/10 → 10/10 | 17.758 → 0.214 | 0.00646 → 0.00822 |
| 1Y–2Y negative row 10745 | 9/10 → 10/10 | 0.555 → 0.456 | 0.00103 → 0.00103 |
| 1Y–2Y positive row 1100 | 7/10 → 10/10 | 4.117 → 0.452 | 0.02039 → 0.01923 |
| One-day negative corner | 7/10 → 7/10 | 38.234 → 20.614 | 1.07604 → 0.14975 |
| One-day positive corner | 7/10 → 7/10 | 33.298 → 18.821 | 0.83376 → 0.13045 |
| Two-year high-vol positive corner | 0/10 → 10/10 | 5.608 → 0.437 | 0.03088 → 0.03148 |

All 81 points now pass both domain checks. In particular, the last corner's
X-domain change fell from 119.701 bp to approximately 0.0000136 bp. Its spatial
change decreased from 1.729916 bp on the preceding refinement to 0.436996 bp,
a ratio of about 3.96, consistent with second-order spatial convergence.

The remaining failed strike indices (one-based) are:

- Negative row 3571: 9 and 10, with spatial changes 10.410 and 27.282 bp.
  Strike 9 now passes the time check; strike 10 still changes by 0.25581 bp.
- One-day negative corner: 8, 9 and 10, with spatial changes 2.928, 8.856 and
  20.614 bp. The last two also have negative coarse-grid call prices.
- One-day positive corner: 1, 2 and 3, with spatial changes 18.821, 8.446 and
  2.880 bp. The first two also have negative coarse-grid put prices. These
  are direct OTM puts, so their failure is unrelated to the positive-call
  integration tail issue that motivated parity.

All eight have finite finest-grid IV estimates, but **none passes the spatial
threshold**. Therefore simply ignoring the coarse-grid negative prices would
not make them acceptable. Their stored accepted IVs remain NaN. The observed
price-change ratios are approximately 3.78–4.24, supporting a spatial truncation
error interpretation; small wing Vega amplifies these price errors in IV units.
For example, row 3571's tenth strike changes in price by only 2.72e-6, yet changes
in IV by 27.28 bp. This is not evidence of a floating-point precision limit or
proof that no other numerical errors remain.

### Measured validation costs

| Case | Old fine solve (s) | New fine solve (s) | New seven-check total (s) |
|---|---:|---:|---:|
| Published example | 195.76 | 357.60 | 799.40 |
| Negative row 3571 | 173.20 | 190.89 | 535.56 |
| Positive row 44913 | 86.54 | 138.44 | 300.26 |
| Negative row 10745 | 171.38 | 199.53 | 502.78 |
| Positive row 1100 | 63.92 | 409.81 | 799.12 |
| One-day negative | 202.53 | 359.70 | 883.80 |
| One-day positive | 62.84 | 284.44 | 525.34 |
| Two-year high-vol positive | 63.87 | 353.78 | 1428.95 |

These are the actual recorded wall times from the convergence experiments,
including concurrent-process contention. They show the cost paid for these
grid choices, but cannot establish a controlled speedup or slowdown factor.
The sum of fine-run wall times is 2294.18 s, and the sum of all final validation
run wall times is 5775.21 s; **neither is the elapsed time of the parallel
experiment**. Exploratory and rejected trials are additional work and are not
included in these sums. Serial paired coarse-grid measurements are recorded
separately in `timing.csv` and `timing.json`.

### Implications for the LN comparison

Recovering previously rejected FDM points changes the comparison population.
On the original 55 accepted points, the new LN/FDM mean absolute difference is
1.098 IV bp, close to the previous 1.108 bp. Across all 72 newly accepted points
with finite LN IVs, it is 13.003 bp; the 95th percentile is 55.388 bp and the
maximum is 297.882 bp. The additional high-volatility corner drives this increase.

That corner has nine finite LN labels, with a maximum LN/FDM difference of
297.882 bp; its tenth LN label remains NaN while FDM passes its checks. These
are **method discrepancies**, not an exact-model error certificate. The corner
still merits an independent pricing benchmark before treating FDM as an oracle
for judging LN accuracy. The small, deliberately selected pilot is not an estimate
of error across the complete training distribution.

The source passes 32 automated tests, including independent matrix-exponential
checks for both fitted and unfitted operators. `reporting_check.json` records
an additional replay of the completed reporting pipeline: 56 recorded grid
results, 81 identical CSV rows, unchanged baseline/source fingerprints, and
preserved failed-label NaNs. That replay does not rerun the PDE or retrain ANN.

The published example's finest IVs still differ from the paper's rounded FDM
column by up to approximately 8.016 bp. Passing 11/11 numerical checks does not
mean that the printed FDM column has been reproduced exactly.

### Serial paired coarse-grid timings

After all numerical workers stopped, the eight cases were priced serially twice
under each configuration's coarse settings. Mean wall seconds per case were:

| Case | Baseline coarse (s) | Refined configuration's coarse (s) |
|---|---:|---:|
| Published example | 2.68 | 4.30 |
| Negative row 3571 | 2.39 | 10.71 |
| Positive row 44913 | 0.64 | 0.76 |
| Negative row 10745 | 4.39 | 4.90 |
| Positive row 1100 | 1.16 | 5.37 |
| One-day negative | 2.99 | 6.71 |
| One-day positive | 0.81 | 1.87 |
| Two-year high-vol positive | 0.70 | 12.56 |
| Sum for one eight-case batch | **15.75** | **47.16** |

The mean CPU-time sums were 15.30 and 45.69 seconds. These two repetitions are
descriptive measurements on this machine, not statistical performance estimates.
The refined configurations generally cost more because they use different grids,
focus parameters and/or time-step counts; this table is not evidence that the
solver engine itself became slower. All re-evaluated coarse prices agree with
their recorded validation prices to within 2.35e-13, and all original-baseline
price identity checks passed. The benchmark did not alter the accepted labels.
