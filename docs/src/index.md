```@meta
CurrentModule = PSIS
```

# PSIS

PSIS.jl implements the Pareto smoothed importance sampling (PSIS) algorithm from [^VehtariSimpson2021].

Given a set of importance weights used in some estimator, PSIS both improves the reliability of the estimates by smoothing the importance weights and acts as a diagnostic of the reliability of the estimates.

See [`psis`](@ref) for details.

## Example

In this example, we use PSIS to smooth log importance ratios for importance sampling 30 isotropic Student ``t``-distributed parameters using standard normal distributions as proposals.

```@example 1
using Random # hide
Random.seed!(42) # hide
using PSIS, Distributions
proposal = Normal()
target = TDist(7)
x = rand(proposal, 1_000, 30)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
nothing # hide
```

```@example 1
result # hide
```

As indicated by the warnings, this is a poor choice of a proposal distribution, and estimates are unlikely to converge (see [`PSISResult`](@ref) for an explanation of the shape thresholds).

When running PSIS with many parameters, it is useful to plot the Pareto shape values to diagnose convergence.
See [Plotting PSIS results](@ref) for examples.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
