"""
    paretoshapeplot(values; kwargs...)
    paretoshapeplot!(values; kwargs...)

Plot shape parameters of fitted Pareto tail distributions for diagnosing convergence.

# Arguments

  - `values`: may be either a vector of Pareto shape parameters or a [`PSISResult`](@ref).

# Keywords

  - `showlines=false`: if `true`, plot horizontal lines indicating relevant Pareto shape
    thresholds are drawn. See [`PSISResult`](@ref) for explanation of thresholds.
  - `backend::Symbol`: backend to use for plotting, defaulting to `:Plots`, unless `:Makie` is
    available.

All remaining keywords are passed to the plotting backend.

See [`psis`](@ref), [`PSISResult`](@ref).

!!! note
    
    Plots.jl or a Makie.jl backend must be loaded to use these functions.

# Examples

```julia
using PSIS, Distributions, Plots
proposal = Normal()
target = TDist(7)
x = rand(proposal, 100, 1_000)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
```

Plot with Plots.jl.

```julia
using Plots
plot(result; showlines=true)
```

Plot with GLMakie.jl.

```julia
using GLMakie
plot(result; showlines=true)
```
"""
paretoshapeplot, paretoshapeplot!

"""
    plotting_backend!(backend::Symbol)

Set default plotting backend. Valid values are `:Plots` and `:Makie`.
"""
plotting_backend!(backend::Symbol) = Base.setindex!(PLOTTING_BACKEND, backend)

function paretoshapeplot(args...; backend=PLOTTING_BACKEND[], kw...)
    return _paretoshapeplot(Val(backend), args...; kw...)
end
function paretoshapeplot!(args...; backend=PLOTTING_BACKEND[], kw...)
    return _paretoshapeplot!(Val(backend), args...; kw...)
end
