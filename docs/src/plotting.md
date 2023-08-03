# Plotting PSIS results

PSIS.jl includes plotting recipes for [`PSISResult`](@ref) using any Plots.jl backend, as well as the utility plotting function [`PSISPlots.paretoshapeplot`](@ref).

We demonstrate this with a simple example.

```@example 1
using Random # hide
Random.seed!(42) # hide
using PSIS, Distributions
proposal = Normal()
target = TDist(7)
ndraws, nchains, nparams = (1_000, 1, 20)
x = rand(proposal, ndraws, nchains, nparams)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
```

## Plots.jl

`PSISResult` objects can be plotted directly:

```@example 1
using Plots
plot(result; showlines=true, marker=:+, legend=false, linewidth=2)
```

This is equivalent to calling `PSISPlots.paretoshapeplot(result; kwargs...)`.
