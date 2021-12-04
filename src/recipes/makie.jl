# define an internal plotting recipe for shape values
@recipe(ParetoShapePlot, pareto_shape) do scene
    l_theme = default_theme(scene, LineSegments)
    return Attributes(;
        default_theme(scene, Scatter)...,
        linewidth=l_theme.linewidth,
        linecolor=(:grey, 0.7),
        linevisible=false,
    )
end

function Makie.plot!(p::ParetoShapePlot)
    attrs = p.attributes
    points = p[1]
    lattr = (:linewidth, :linecolor, :linevisible)
    linewidth, linecolor, linevisible = getindex.(p, lattr)
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
            visible=linevisible,
        )
    end
    ks = filter(∉(lattr), keys(attrs))
    return scatter!(p, p[1]; (k => attrs[k] for k in ks)...)
end

# designate default plot type for PSISResult
Makie.plottype(::PSISResult) = ParetoShapePlot

# convert PSISResult to its shape values with paretoshapeplot
function Makie.convert_arguments(::Type{<:ParetoShapePlot}, result::PSISResult)
    return convert_arguments(Scatter, as_array(missing_to_nan(pareto_shape(result))))
end
