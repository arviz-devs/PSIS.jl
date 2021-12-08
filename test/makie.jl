using PSIS
using GLMakie
using Test

@testset "Makie.jl recipes" begin
    PSIS.plotting_backend!(:Makie)

    @testset "paretoshapeplot/plot" begin
        @testset "$f(::$T; showlines=$showlines) sz=$sz" for f in
                                                             (paretoshapeplot, Makie.plot),
            T in (PSISResult, Vector),
            showlines in (true, false),
            sz in (100, (10, 100))

            f === Makie.plot && T <: Vector && continue
            result = psis(randn(sz...))
            values = T <: PSISResult ? result : result.pareto_shape
            plt = f(values; showlines=showlines, linewidth=5)
            @test plt.plot isa PSIS.MakieRecipe.ParetoShapePlot
            plots = plt.plot.plots
            @test plt.axis.xlabel[] == "Parameter index"
            @test plt.axis.ylabel[] == "Pareto shape"
            @test length(plots) == 5
            @test plots[5] isa Scatter
            @test plots[5].converted[1][] ==
                Point2f0.(Base.OneTo(result.nparams), PSIS.as_array(result.pareto_shape))
            linestyles = [:dot, :dashdot, :dash, :solid]
            yvals = [0, 0.5, 0.7, 1]
            line_xmin, line_xmax = sz isa Tuple ? (0.91, 10.09) : (1, 1)
            for i in 1:4
                @test plots[i] isa LineSegments
                @test plots[i].converted[1][] ==
                    [Point2f0(line_xmin, yvals[i]), Point2f0(line_xmax, yvals[i])]
                @test plots[i].attributes[:linestyle][] == linestyles[i]
                @test plots[i].attributes[:linewidth][] == 5
                @test plots[i].attributes[:visible][] == showlines
            end
        end
    end

    @testset "paretoshapeplot!/plot!" begin
        @testset "$f(::$T; showlines=$showlines) sz=$sz" for f! in (
                paretoshapeplot!, Makie.plot!
            ),
            T in (PSISResult, Vector),
            showlines in (true, false),
            sz in (100, (10, 100))

            f! === Makie.plot! && T <: Vector && continue
            result = psis(randn(sz...))
            values = T <: PSISResult ? result : result.pareto_shape
            fig = Figure()
            ax = Axis(fig[1, 1])
            plt = f!(values; showlines=showlines, linewidth=5)
            axis = fig.content[1]
            @test last(axis.scene.plots) isa PSIS.MakieRecipe.ParetoShapePlot
            plots = last(axis.scene.plots).plots
            @test axis.xlabel[] == "Parameter index"
            @test axis.ylabel[] == "Pareto shape"
            @test length(plots) == 5
            @test plots[5] isa Scatter
            @test plots[5].converted[1][] ==
                Point2f0.(Base.OneTo(result.nparams), PSIS.as_array(result.pareto_shape))
            linestyles = [:dot, :dashdot, :dash, :solid]
            yvals = [0, 0.5, 0.7, 1]
            line_xmin, line_xmax = sz isa Tuple ? (0.91, 10.09) : (1, 1)
            for i in 1:4
                @test plots[i] isa LineSegments
                @test plots[i].converted[1][] ==
                    [Point2f0(line_xmin, yvals[i]), Point2f0(line_xmax, yvals[i])]
                @test plots[i].attributes[:linestyle][] == linestyles[i]
                @test plots[i].attributes[:linewidth][] == 5
                @test plots[i].attributes[:visible][] == showlines
            end
        end
    end
end
