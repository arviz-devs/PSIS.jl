"""
    paretoshapeplot(values; showlines=false)
    paretoshapeplot!(values; showlines=false)

Plot shape parameters of fitted Pareto tail distributions for diagnosing convergence.

`values` may be either a vector of Pareto shape parameters or a [`PSISResult`](@ref).

If `showlines=true`, then horizontal lines indicating the different Pareto shape thresholds
are drawn.

See [`psis`](@ref), [`PSISResult`](@ref).

!!! note
    
    Plots.jl must be loaded to use these functions.

## Example

```julia
using PSIS, Distributions, Plots
proposal = Normal()
target = TDist(7)
x = rand(proposal, 100, 1_000)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
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
