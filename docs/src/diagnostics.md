```@meta
CurrentModule = PSIS
```

# Diagnosing and smoothing tails

Pareto diagnostics can be computed without changing observations or calculating
importance weights. See [`pareto_khat`](@ref) for their interpretation alongside
MCMC diagnostics.

```@example diagnostics
using PSIS, Distributions, Random
rng = Xoshiro(42)
x = rand(rng, TDist(3), 1000, 4)
pareto_diagnose(x)
```

The first two dimensions are draws and chains. Additional dimensions describe
variables. `pareto_khat(x)` returns just the largest of the fitted left/right
tail shapes. `tail=:left` and `tail=:right` select one tail. `pareto_diagnose`
returns a `ParetoDiagnostics` record containing:

- `khat`: tail shape; see [`pareto_khat`](@ref);
- `khat_threshold`: sample-size threshold; see [`pareto_khat_threshold`](@ref);
- `min_ss`: suggested sample size; see [`pareto_min_ss`](@ref);
- `convergence_rate`: relative RMSE rate; see [`pareto_convergence_rate`](@ref).

The linked docstrings give the interpretation, assumptions, and limitations
of each diagnostic.

## Relative efficiency and summaries

`r_eff=1` assumes independent draws. PSIS does not automatically estimate MCMC
autocorrelation. For a tail diagnostic, supply a tail-efficiency estimate:

```@example diagnostics
using MCMCDiagnosticTools
r_eff = ess(x; kind=:tail, relative=true)
pareto_diagnose(x; r_eff)
```

Scalar diagnostic functions can be passed to `PosteriorStats.summarize`. No
dependency on PosteriorStats is required in PSIS.

```julia
using PosteriorStats, PSIS, Statistics, MCMCDiagnosticTools
summarize(draws, mean, std, pareto_khat)

# Explicitly account for MCMC dependence, separately for each variable:
mcmc_pareto_khat(x) = pareto_khat(x; r_eff=ess(x; kind=:tail, relative=true))
summarize(draws, :pareto_khat => mcmc_pareto_khat)
```

For several diagnostics at once, `pareto_diagnose` reuses a single fit per tail.

## Diagnostics from an existing fit

The same functions accept scalar fitted shapes or sample sizes. Array arguments
always represent observations, so broadcast when working with arrays of k-hats:

```@example diagnostics
d = pareto_diagnose(x)
n = size(x, 1) * size(x, 2) # pooled draws and chains
pareto_min_ss(d.khat)
pareto_khat_threshold(n)
pareto_convergence_rate(d.khat, n)
```

```@example diagnostics
khats = [0.3, 0.5, 0.7]
pareto_min_ss.(khats)
pareto_convergence_rate.(khats, n)
```

## Smoothing

```@example diagnostics
smoothed, diagnostics = pareto_smooth(x)
(original_mean=mean(x), smoothed_mean=mean(smoothed), diagnostics)
```

Smoothing returns a new array with the same shape and order. It replaces only
the selected tails, caps replacements at the observed extremes, and does not
normalize the observations. Diagnostics describe the original tails.

Use `psis(log_ratios)` for importance weights. Its `PSISResult` retains the
fitted k-hat; `pareto_diagnose(result)` calculates the additional diagnostics
without fitting the smoothed weights again.

## Log scale and exceptional inputs

`is_log_scale=true` treats the input as log-encoded positive observations and
fits their original-scale right tail, with stable scaling to avoid overflow.
This is different from diagnosing the expectation of a log-transformed variable.
The log-encoded mode supports only `tail=:right` and selects it by default.

The default tail count is `floor(min(n/5, 3sqrt(n/r_eff)))`. A `tail_length`
override must leave a cutoff observation and nonoverlapping tails. Fits need at
least five tail observations. Constant samples, constant tails, nonfinite data,
and insufficient observations return `NaN` k-hat. A failed tail is not smoothed.
If either requested tail fails, the combined diagnostic is `NaN`.

Negative infinity denotes zero in log-encoded data and is allowed outside the
fitted tail, including at the cutoff. Positive infinity and NaN are invalid.
Invalid options or relative efficiencies raise errors. Diagnostic and generic
smoothing functions are quiet unless `warn=true`; `psis` warns by default.

## Changes from v0.9

The v0.10 API uses `r_eff` and `pareto_khat`, and `PSISResult` contains only
`log_weights`, `pareto_khat`, and `ess_is`. Weights are unnormalized; use
`LogExpFunctions.softmax` along sample dimensions when normalized weights are
needed. The computed properties, public `psis!` and `ess_is` functions, and plot
recipe were removed in the v0.10 API simplification.

This redesign additionally uses floor-based tail counts from the v9 paper,
sample-size-aware reliability warnings and display, and `NaN` ESS for unreliable
or failed fits. Invalid `r_eff` now raises `ArgumentError`. The existing GPD
fitter is retained. Expectation-specific diagnostics are reserved for a later
iteration.
