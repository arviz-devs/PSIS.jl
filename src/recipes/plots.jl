# Plots.jl recipes
module PSISPlots

using ..PSIS
using RecipesBase: RecipesBase

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
