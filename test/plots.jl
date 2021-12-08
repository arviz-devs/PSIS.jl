using PSIS
using Plots
using Test

@testset "Plots.jl recipes" begin
    PSIS.plotting_backend!(:Plots)

    @testset "$f" for f in (paretoshapeplot, paretoshapeplot!), sz in (100, (10, 100))
        result = psis(randn(sz...))

        @testset for values in (result, result.pareto_shape)
            plot()
            plt = f(values)
            @test plt isa Plots.Plot
            @test length(plt.series_list) == 1
            @test plt[1][1][:x] == Base.OneTo(result.nparams)
            @test plt[1][1][:y] == PSIS.as_array(result.pareto_shape)
            @test plt[1][1][:seriestype] == :scatter
            @test plt[1][:xaxis][:guide] == "Parameter index"
            @test plt[1][:yaxis][:guide] == "Pareto shape"
        end

        plot()
        plt2 = f(result; showlines=true)
        @test plt2 isa Plots.Plot
        @test length(plt2.series_list) == 5
        linestyles = [:dot, :dashdot, :dash, :solid]
        yvals = [0, 0.5, 0.7, 1]
        for i in 1:4
            @test plt2[1][i][:y] == fill(yvals[i], 3)
            @test plt2[1][i][:linestyle] == linestyles[i]
            @test plt2[1][i][:linecolor] ==
                RGBA(0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 1.0)
            @test plt2[1][i][:linealpha] == 0.7
        end
    end

    @testset "plot(::PSISResult)" begin
        result = psis(randn(10, 100))
        for showlines in (true, false)
            plt = paretoshapeplot(result; showlines=showlines)
            plt2 = plot(result; showlines=showlines)
            @test length(plt.series_list) == length(plt2.series_list)
            for (s1, s2) in zip(plt.series_list, plt2.series_list)
                @test s1[:seriestype] === s2[:seriestype]
                @test isequal(s1[:x], s2[:x])
                @test isequal(s1[:y], s2[:y])
            end
        end
    end

    @testset "plot(::PSISResult; seriestype=:path)" begin
        result = psis(randn(10, 100))
        plt = plot(result; seriestype=:path)
        @test length(plt.series_list) == 1
        @test plt[1][1][:x] == eachindex(result.pareto_shape)
        @test plt[1][1][:y] == result.pareto_shape
        @test plt[1][1][:seriestype] == :path
    end
end
