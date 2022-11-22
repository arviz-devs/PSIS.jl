# Plots.jl recipes

"""
A module defining [`paretoshapeplot`](@ref) for plotting Pareto shape values with Plots.jl
"""
module PSISPlots

using ..PSIS
using RecipesBase: RecipesBase

"""
    paretoshapeplot(values; showlines=false, ...)
    paretoshapeplot!(values; showlines=false, kwargs...)

Plot shape parameters of fitted Pareto tail distributions for diagnosing convergence.

`values` may be either a vector of Pareto shape parameters or a [`PSIS.PSISResult`](@ref).

If `showlines==true`, horizontal lines indicating relevant Pareto shape thresholds are
drawn. See [`PSIS.PSISResult`](@ref) for an explanation of the thresholds.

All remaining `kwargs` are forwarded to the plotting function.

See [`psis`](@ref), [`PSISResult`](@ref).

# Examples

```julia
using PSIS, Distributions, Plots
proposal = Normal()
target = TDist(7)
x = rand(proposal, 100, 1_000)
log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
result = psis(log_ratios)
paretoshapeplot(result)
```

We can also plot the Pareto shape parameters directly:

```julia
paretoshapeplot(result.pareto_shape)
```

We can also use `plot` directly:

```julia
plot(result.pareto_shape; showlines=true)
```
"""
paretoshapeplot, paretoshapeplot!

RecipesBase.@userplot ParetoShapePlot

RecipesBase.@recipe function f(plt::ParetoShapePlot; showlines=false)
    showlines && RecipesBase.@series begin
        seriestype := :hline
        primary := false
        linestyle --> [:dot :dashdot :dash :solid]
        linealpha --> 0.7
        linecolor --> :grey
        y := [0 0.5 0.7 1]
    end
    title --> ""  # no title unless specified by the user
    ylabel --> "Pareto shape"
    seriestype --> :scatter
    arg = first(plt.args)
    k = arg isa PSIS.PSISResult ? PSIS.pareto_shape(arg) : arg
    return (PSIS.as_array(PSIS.missing_to_nan(k)),)
end

# plot PSISResult using paretoshapeplot if seriestype not specified
RecipesBase.@recipe function f(result::PSISResult)
    if haskey(plotattributes, :seriestype)
        k = PSIS.as_array(PSIS.missing_to_nan(PSIS.pareto_shape(result)))
        return (k,)
    else
        return ParetoShapePlot((result,))
    end
end

end # module PSISPlots
