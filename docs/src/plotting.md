# Plotting PSIS results

PSIS.jl includes plotting recipes for [`PSISResult`](@ref) using any Plots.jl or Makie.jl backend, as well as the utility plotting function [`paretoshapeplot`](@ref).

We demonstrate this with a simple example.

```@example 1
using Random # hide
Random.seed!(42) # hide
using PSIS, Distributions
proposal = Normal()
target = TDist(7)
x = rand(proposal, 20, 1_000)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
```

## Plots.jl

`PSISResult` objects can be plotted directly:

```@example 1
using Plots
Plots.plot(result; showlines=true, marker=:+, legend=false, linewidth=2)
```

This is equivalent to calling `paretoshapeplot(result; kwargs...)`.

## Makie.jl

The same syntax is supported with Makie.jl backends.

```@example 1
using CairoMakie
CairoMakie.activate!(; type = "svg") # hide
Makie.inline!(true) # hide
Makie.plot(result; showlines=true, marker=:+)
```

## Selecting the backend

If a Makie backend is loaded, then by default `paretoshapeplot` and `paretoshapeplot!` will use that backend; otherwise, a Plots.jl backend is used if available.
However, both functions accept a `backend` keyword that can be used to specify the backend if for some reason both are loaded.

```@example 1
paretoshapeplot(result; backend=:Plots)
```

```@example 1
paretoshapeplot(result; backend=:Makie)
```
