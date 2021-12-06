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
RecipesBase.@recipe function f(config::ParetoShapePlotConfig; showlines=false)
    showlines && RecipesBase.@series begin
        seriestype := :hline
        primary := false
        linestyle --> [:dot :dashdot :dash :solid]
        linealpha --> 0.7
        linecolor --> :grey
        y := [0 0.5 0.7 1]
    end
    xlabel --> "Parameter index"
    ylabel --> "Pareto shape"
    seriestype --> :scatter
    arg = first(config.args)
    ξ = as_array(missing_to_nan(arg isa PSISResult ? pareto_shape(arg) : arg))
    return (ξ,)
end

# plot PSISResult using paretoshapeplot if seriestype not specified
RecipesBase.@recipe function f(result::PSISResult)
    if haskey(plotattributes, :seriestype)
        ξ = as_array(missing_to_nan(pareto_shape(result)))
        return (ξ,)
    else
        return ParetoShapePlotConfig((result,))
    end
end
