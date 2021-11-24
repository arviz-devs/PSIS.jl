# type recipe, just maps PSISResult to shape(s)
RecipesBase.@recipe function f(::Type{T}, result::T) where {T<:PSISResult}
    return pareto_shape(result)
end

# user recipe, plots PSISResult with lines
@recipe function f(result::PSISResult; show_hlines=false)
    show_hlines && @series begin
        seriestype := :hline
        primary := false
        linestyle --> [:dot :dashdot :dash :solid]
        linealpha --> 0.7
        y := [0 0.5 0.7 1]
    end
    ξ = pareto_shape(result)
    ξnew = if ξ isa Missing
        [NaN]
    elseif ξ isa Number
        [ξ]
    else
        replace(ξ, missing => NaN)
    end
    seriestype --> :scatter
    markershape --> :cross
    primary := true
    ylabel --> "Pareto shape"
    xlabel --> "Parameter"
    return (ξnew,)
end
