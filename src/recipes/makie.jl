# Makie.jl recipes

# if Makie is loaded, use Makie by default
PLOTTING_BACKEND[] = :Makie

# encase the recipe in a module to avoid namespace pollution
module MakieRecipe

using ..Makie

# define an internal plotting recipe for shape values
@recipe(ParetoShapePlot, pareto_shape) do scene
    l_theme = default_theme(scene, LineSegments)
    return Attributes(;
        default_theme(scene, Scatter)...,
        linewidth=l_theme.linewidth,
        linecolor=(:grey, 0.7),
        showlines=false,
    )
end

end # module

using .MakieRecipe: ParetoShapePlot

# main plotting function
function _paretoshapeplot(backend::Val{:Makie}, args...; attributes...)
    return Makie.plot(ParetoShapePlot, args...; attributes...)
end
function _paretoshapeplot!(backend::Val{:Makie}, args...; attributes...)
    return Makie.plot!(ParetoShapePlot, args...; attributes...)
end

function Makie.plot!(p::ParetoShapePlot)
    attrs = p.attributes
    points = p[1]
    lattr = (:linewidth, :linecolor, :showlines)
    linewidth, linecolor, showlines = getindex.(p, lattr)
    xminmax = lift(points) do pts
        xmin, xmax = extrema(first, pts)
        xspan = xmax - xmin
        return [xmin - 0.01 * xspan, xmax + 0.01 * xspan]
    end
    thresholds = (0, 0.5, 0.7, 1)
    linestyles = (:dot, :dashdot, :dash, :solid)
    for (thresh, linestyle) in zip(thresholds, linestyles)
        linesegments!(
            p,
            xminmax,
            fill(thresh, 2);
            linestyle=linestyle,
            linewidth=linewidth,
            color=linecolor,
            visible=showlines,
        )
    end
    ks = filter(∉(lattr), keys(attrs))
    return scatter!(p, p[1]; (k => attrs[k] for k in ks)...)
end

# designate default plot type for PSISResult
Makie.plottype(::PSISResult) = ParetoShapePlot

# convert PSISResult to its shape values when plotted
function Makie.convert_arguments(P::Makie.PlotFunc, result::PSISResult)
    return convert_arguments(P, as_array(missing_to_nan(pareto_shape(result))))
end
function Makie.convert_arguments(
    ::Type{<:ParetoShapePlot}, pareto_shape::Union{Real,AbstractVector{<:Real}}
)
    return convert_arguments(Scatter, as_array(pareto_shape))
end

# set default labels when an Axis is available
function Makie.plot!(ax::Axis, P::Type{<:ParetoShapePlot}, attributes::Attributes, args...)
    plt = invoke(
        Makie.plot!,
        Tuple{typeof(ax),Makie.PlotFunc,typeof(attributes),typeof.(args)...},
        ax,
        P,
        attributes,
        args...,
    )
    _set_default_makie_axis_labels!(ax)
    return plt
end
# dispatch on cases like plot(::PSISResult)
function Makie.plot!(ax::Axis, P::Type{Any}, attributes::Attributes, result::PSISResult)
    plt = Makie.plot!(ax, ParetoShapePlot, attributes, result)
    _set_default_makie_axis_labels!(ax)
    return plt
end

function _set_default_makie_axis_labels!(ax)
    if isempty(ax.xlabel[])
        ax.xlabel = "Parameter index"
    end
    if isempty(ax.ylabel[])
        ax.ylabel = "Pareto shape"
    end
    return ax
end
