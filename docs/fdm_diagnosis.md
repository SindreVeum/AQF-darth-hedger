# Figure 7 FDM discrepancy investigation

## Scope and conclusion

This investigation retains the published Figure 7 parameters: F=1, beta=1,
T=540/365, alpha=0.30, nu=0.60, rho=-0.35, and all 50 recovered PDF strike
positions. Calls and puts are priced directly. No parity conversion, SLN,
training or full-dataset generation is involved. The production solver and the
previous finest-grid results have not been changed to fit the paper.

The checks so far support our converged price level. They do not identify an
algebraic or Craig-Sneyd implementation error capable of explaining 31.79 IV bp.
Independent simulation and a different PDE discretization provide additional
evidence beyond self-convergence. However, the paper does not disclose enough
numerical settings to identify the exact source of its FDM curve's discrepancy.
An author-side implementation error and undisclosed numerical choices cannot
be distinguished conclusively without the original implementation.

## What the 31.79 bp measures

One IV bp is 0.0001 annual decimal Black volatility. One price bp here is 0.0001
undiscounted price units, with F=1. These units must not be interchanged.

| Strike K | Our FDM IV (%) | Paper FDM IV (%) | Our minus paper (IV bp) | Our minus paper (price bp) |
|---|---:|---:|---:|---:|
| 0.1107358512 | 64.3754255 | 64.0575014 | +31.792408 | +0.089885 |
| 1.2647478122 | 28.8211183 | 29.0089227 | -18.780443 | -8.074371 |

Figure 7 has no separately published price table. The paper prices in this
investigation are Black prices inferred from its digitized IV curve.

At the first strike the OTM put has very low Vega, approximately 0.00288194.
The small price difference is amplified by IV inversion:

$$\Delta\sigma_{\rm IV}\simeq\frac{\Delta V}{\operatorname{Vega}}.$$

This explains the scale of the first IV difference, not its numerical cause.
The discrepancy is not confined to an economically tiny wing price: over all
50 points, the price-difference MAE is 1.11372 bp and its maximum is 8.07437 bp.
Both sets of recovered call prices pass discrete convexity on these strikes;
a jagged IV curve alone does not establish arbitrage or a solver bug.

## Reference and formula audit

- The independent Hagan reconstruction matches its plotted curve within
  0.021902 IV bp, and our LN matches the plotted LN within 0.144395 IV bp.
- PDF vertical-coordinate rounding is about 0.01594 IV bp. This is a drawing
  precision estimate, not an estimate of the author's numerical accuracy.
- The 2018 preprint's red raster curve is consistent with the journal's 50
  recovered FDM vertices: all project within 0.8 raster pixels of the red line.
- Section 3.3 specifies log-forward/log-volatility coordinates and Craig-Sneyd.
  Table 8 reports 400x200x100 but does not assign the three counts to axes.
  Domain bounds, boundary conditions, grid spacing, payoff treatment and
  interpolation are not disclosed sufficiently to reconstruct the author's grid.

The implemented transformed PDE is

$$u_\tau=\tfrac12\alpha^2 e^{2y}(u_{xx}-u_x)
 +\tfrac12\nu^2(u_{yy}-u_y)+\rho\nu\alpha e^y u_{xy},$$

with x=log(F/K), y=log(sigma/alpha), and V=K u. The transformation, mixed
derivative coefficient/sign, nonuniform derivative weights and original
Craig-Sneyd correction were audited. The existing matrix-exponential tests
check time integration independently of the ADI stages. Exact Black controls
check the deterministic-volatility limit.

## Controlled grid, domain and interpolation experiments

The nominal experiment interprets 400x200x100 as X intervals, Y intervals and
time steps. That assignment is an explicit hypothesis, not a recovered author
setting. The nominal grid has half-widths 8 and 6, sinh focus parameters 0.5 and
1, centered payoff-cell averaging, and cubic price interpolation.

| Variant | Max change versus nominal (IV bp) | Max difference versus paper (IV bp) |
|---|---:|---:|
| Nominal sinh grid | 0 | 32.865897 |
| Uniform log-coordinate grid, same counts/bounds | 2.779470 | 35.061269 |
| X half-width reduced to 3.01296 | 6.144232 | 28.739198 |
| Y half-width reduced to 3.00175 | 0.012638 | 32.853258 |
| Y half-width reduced to 1.50233 | 307.605569 | 274.739672 |

Domain contractions preserve every remaining interior node, so their differences
isolate the moved boundary rather than a simultaneous change in grid spacing.
Linear instead of cubic interpolation on the **same evolved surface** changes
IV by up to 0.588708 bp on the nominal sinh grid and 5.788776 bp on the uniform
grid. None of these configurations reproduces the complete paper FDM curve.

The saved finest-grid experiment is already stable under further domain
expansion: maximum X/Y changes are 0.000207/2.50e-11 IV bp on its medium grid.
Its last spatial refinement changes IV by at most 0.473972 bp and time
refinement by 0.009687 bp. These diagnostics support the present wide domain;
they do not show that every possible finite boundary choice is harmless.

### Common log-forward grid and payoff alignment

The production grid uses x=log(F/K) and centers the payoff kink at x=0. A
common log(F) grid instead puts each strike's kink at log(K), at different
positions relative to the nodes. This is a testable numerical mechanism, not
an established description of the author's code.

A diagnostic retains the same CS stages and spatial operators, changing only
payoff initialization and readout in memory. It evolves the 50 direct OTM
payoffs on a common uniform log(F) grid with 400 X intervals, 200 sinh Y
intervals, 100 time steps and widths 8/6, reading F=1 exactly at a node.
With raw nodal payoffs, its maximum difference from our fine result is
2.724386 IV bp. Its maximum difference from the paper is still 32.923701 bp;
at K=1.2647478 the difference is -19.752325 bp. This tested alignment setting
does not explain the paper curve. Because both centering and X spacing differ
from the baseline, the comparison must not be attributed solely to smoothing.
Replacing raw payoffs by analytical kink-cell averages at four selected
strikes changes IV by 0.05363 to 3.06168 bp, without uniformly improving
accuracy; the first-strike gap against the paper increases to 33.356971 bp.

The guarded diagnostic is archived in `archive/fdm/fdm_fixed_grid_check.py`, with raw results
and deterministic-Black controls in `fixed_grid_check.json`. It never edits
the production solver. Coarse Black wing IVs are very sensitive to tiny price
errors, so these deliberately coarse configurations are not accepted labels.

## Independent conditional simulation

This check does not use the paper's conditional-LN approximation, an FDM
operator or put-call parity. Simulate the volatility GBM exactly at time nodes:

$$\sigma_t=\alpha\exp(\nu W_t-\tfrac12\nu^2t),\qquad
 I=\int_0^T\sigma_t^2\,dt.$$

Conditional on that volatility path, the asset's orthogonal Brownian component
can be integrated analytically. Its conditional Black forward and variance are

$$m=\exp\left[\frac{\rho}{\nu}(\sigma_T-\alpha)
                   -\tfrac12\rho^2 I\right],\qquad v=(1-\rho^2)I.$$

Average the conditional call or put Black price over the simulated paths.
The only path-discretization approximation is the trapezoidal integral I;
grids with 128/256/512 time intervals are nested. Brownian-bridge randomized Sobol sampling uses
24 independent scrambles of 65,536 paths each, 1,572,864 paths in total.

| K | Our FDM minus QMC (IV bp) | Paper FDM minus QMC (IV bp) |
|---|---:|---:|
| 0.1107358512 | +0.35360 | -31.43881 |
| 1.0724231938 | -0.00245 | +11.72634 |
| 1.2647478122 | +0.00123 | +18.78167 |
| 1.9379358763 | -0.02009 | +9.36159 |

At the first strike, QMC IV is 64.3718895%, with an empirical marginal 95%
randomization interval [64.3692116%, 64.3745665%]. Our 64.3754255% is slightly
outside that interval (2.73 observed price standard errors). The remaining
approximately 0.35 IV bp wing discrepancy is unresolved; this evidence does
not certify the current FDM as exact. For context, the last FDM spatial change
at this strike is 0.341196 bp; a second-order Richardson estimate would remove
approximately 0.113732 bp from the current fine result, but is not a rigorous
error bound or a replacement label.

The last nested QMC time changes are +0.00802, -0.000249, -0.0000794 and
+0.000721 IV bp at these four strikes. Such observed changes assess but do not
bound time bias. Student-t intervals are estimated across independent scramble
means, exclude time bias, and are neither deterministic nor simultaneous error
bounds. The second batch was prompted by the residual in the first; both are
retained and pooled, with no selection of favorable results. The lower-path
pilot overlaps the first batch's sequences and is not pooled with them.

Deterministic controls recover Brownian covariance min(s,t) within 4.44e-16
and constant-volatility conditional Black prices within 1.67e-16. See
`mc_validation.json` and `mc_figure7_pooled.json` for seeds, batch results,
uncertainty and exact reproduction commands. Only these four strikes received
the independent simulation check; the conclusion is not a full-domain validation.

## Independent PDE cross-check

A separate solver uses f=F/K, y=log(sigma/alpha), with

$$u_\tau=\tfrac12\alpha^2e^{2y}f^2u_{ff}
 +\tfrac12\nu^2(u_{yy}-u_y)+\rho\nu\alpha e^y f u_{fy}.$$

It assembles a complete sparse operator and uses unsplit Crank-Nicolson with
two initial backward-Euler half steps, rather than the production log-forward
Craig-Sneyd implementation. Both payoffs are evolved directly. The forward
mesh, derivative construction and time integrator are independently implemented.

On the wide f domain (upper bound at least 128), refining n_left/ny from
80/100 to 160/200 and 320/400 produces the following comparisons:

| Independent PDE result | First minus our CS (IV bp) | First minus paper (IV bp) | Max absolute difference from our CS, all 50 (IV bp) |
|---|---:|---:|---:|
| 160/200 grid | +1.427547 | +33.219955 | 3.178833 |
| 320/400 grid | +0.277263 | +32.069671 | 0.650470 |
| Fine/finest Richardson diagnostic | -0.106165 | +31.686244 | 0.192317 |

The last grid has 1,310 forward nodes, 401 Y nodes and 400 time steps. The
first-strike spatial change ratio is 3.967, consistent with second order.
Richardson uses (4*IV_finest-IV_fine)/3, assuming approximately quadratic
leading spatial error. It is diagnostic extrapolation, not an independent
exact price or a replacement label. Its first-strike estimate changes by only
0.012479 bp relative to the preceding two-grid extrapolation.
Applying the same spatial extrapolation diagnostically to our CS grids gives
agreement with the independent PDE extrapolation within 0.034326 IV bp over
all 50 strikes (0.007567 bp at the first). This agreement depends on the
second-order assumptions and does not remove common time/domain errors.

On the coarse wide forward grid, doubling 400 time steps to 800 changes IV
by at most 0.01218 bp. Changing the wide Y-boundary condition from reflecting
to frozen payoff changes results only at roundoff scale. Independent exact
Black controls show second-order spatial convergence: maximum IV errors
1.224, 0.306 and 0.0768 bp under successive mesh doublings.

## Why the markers become denser on the right

Figure 7 and our original comparison already use a logarithmic strike axis.
The recovered strikes are approximately equally spaced in K, with
Delta K=0.096168. Consequently,

$$\Delta\log K=\log(1+\Delta K/K)$$

decreases as K increases. The first log gap is 0.625113 and the last is
0.020145: right-side spacing is about 31 times smaller. This is expected
coordinate geometry, not duplicate samples or a pricing defect.

`results/fdm_diagnosis/strike_spacing.png` shows the same 50 points on linear
and logarithmic axes. A linear-K companion gives approximately equal marker
spacing. Geometrically spaced strikes would be equally spaced on a log axis,
but would change the paper benchmark, so no resampling was performed.

## Reproduction and saved evidence

Run from the repository root:

```sh
python -m archive.fdm.fdm_diagnosis
python -m src.fdm_mc_check --power 16 --repeats 24 --steps 512 --seed 202611260 --output results/fdm_diagnosis/mc_figure7_24_scrambles.json
python -m src.fdm_independent --n-left 160 --ny 200 --nt 400 --f-max 128 --name fine_wide
python -m src.fdm_independent --n-left 320 --ny 400 --nt 400 --f-max 128 --name finest_wide
python -m archive.fdm.fdm_fixed_grid_check --skip-fine
python -m pytest -q
```

`grid_variants.json` and `.csv` contain every strike/result; `source_audit.json`
records source precision and sampling evidence. `validation.json` checks that
observing the PDE slice leaves prices bitwise identical, contracted domains
retain their interior nodes, and the production/baseline hashes remain unchanged.
The independent checks have separate result files. No historical results were
overwritten. Diagnostic timings are observed runtimes, not a controlled speed
comparison: some checks ran concurrently.

The full regression suite passed: 38 tests. The fixed-grid diagnostic completed
four Black controls, one raw-payoff 50-strike run and one averaged-payoff
four-strike run. Optional 800-X-interval SABR runs were not needed or performed.
