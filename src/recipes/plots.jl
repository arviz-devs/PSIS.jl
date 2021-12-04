# Plots.jl recipes

# plot config object, used for dispatch
mutable struct ParetoShapePlotConfig
    args
end
paretoshapeplot(args...; kw...) = RecipesBase.plot(ParetoShapePlotConfig(args); kw...)
paretoshapeplot!(args...; kw...) = RecipesBase.plot!(ParetoShapePlotConfig(args); kw...)
function paretoshapeplot!(plt::RecipesBase.AbstractPlot, args...; kw...)
    return RecipesBase.plot!(plt, ParetoShapePlotConfig(args); kw...)
end

# pre-process to make PSISResult if necessary
RecipesBase.@recipe function f(config::ParetoShapePlotConfig)
    arg = first(config.args)
    if arg isa PSISResult
        return (arg,)
    else
        return (psis(arg),)
    end
end

# plot PSISResult with lines
RecipesBase.@recipe function f(result::PSISResult; showlines=false)
    showlines && RecipesBase.@series begin
        seriestype := :hline
        primary := false
        linestyle --> [:dot :dashdot :dash :solid]
        linealpha --> 0.7
        linecolor --> :grey
        y := [0 0.5 0.7 1]
    end
    ξ = as_array(missing_to_nan(pareto_shape(result)))
    seriestype --> :scatter
    primary := true
    ylabel --> "Pareto shape"
    xlabel --> "Parameter index"
    return (ξ,)
end
