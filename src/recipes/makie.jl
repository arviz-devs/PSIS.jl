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

# convert PSISResult to its shape values with ParetoShapePlot
function Makie.convert_arguments(::Type{<:ParetoShapePlot}, result::PSISResult)
    return convert_arguments(Scatter, as_array(missing_to_nan(pareto_shape(result))))
end

# overloads to set default labels
# Note: these are annoying hacks, since it's not clear the best internal overload to use
# to modify the plot after an axis has been internally created.

function _paretoshapeplot(backend::Val{:Makie}, arg; attributes...)
    plt = Makie.plot(ParetoShapePlot, arg; attributes...)
    return _set_default_makie_labels!(plt)
end
function _paretoshapeplot!(backend::Val{:Makie}, args...; attributes...)
    plt = Makie.plot!(ParetoShapePlot, args...; attributes...)
    return _set_default_makie_labels!(plt)
end
function Makie.plot(result::PSISResult; attributes...)
    plt = plot(ParetoShapePlot, result; attributes...)
    return _set_default_makie_labels!(plt)
end
function Makie.plot!(result::PSISResult; attributes...)
    plt = Makie.plot!(ParetoShapePlot, result; attributes...)
    return _set_default_makie_labels!(plt)
end
function Makie.plot!(layoutable, result::PSISResult; attributes...)
    plt = Makie.plot!(ParetoShapePlot, layoutable, result; attributes...)
    return _set_default_makie_labels!(plt)
end

function _set_default_makie_labels!(plt)
    labels = (xlabel = "Parameter index", ylabel = "Pareto shape")
    for (k, label) in pairs(labels)
        isempty(getproperty(plt.axis, k)[]) && setproperty!(plt.axis, k, label)
    end
    return plt
end
