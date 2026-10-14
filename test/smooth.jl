using PSIS
using Test
using StableRNGs
using Distributions: TDist
using DimensionalData: DimArray, Dimensions

@testset "generic Pareto smoothing" begin
    rng = StableRNG(33)
    x = rand(rng, TDist(3), 1000, 2)
    original = copy(x)
    for tail in (:left, :right, :both)
        smoothed, d = @inferred pareto_smooth(x; tail)
        @test x == original
        @test size(smoothed) == size(x)
        @test smoothed != x
        @test d == pareto_diagnose(x; tail)
        @test minimum(smoothed) >= minimum(x)
        @test maximum(smoothed) <= maximum(x)
        perm = sortperm(vec(x))
        @test issorted(smoothed[perm])
        @test smoothed[perm[135:1866]] == x[perm[135:1866]]
        @test sum(smoothed) != 1
    end
    left, _ = pareto_smooth(-x; tail=:left)
    right, _ = pareto_smooth(x; tail=:right)
    @test left ≈ -right
    positive = abs.(x)
    raw, rawdiag = pareto_smooth(positive; tail=:right)
    logsmoothed, logdiag = pareto_smooth(log.(positive); is_log_scale=true)
    @test exp.(logsmoothed) ≈ raw
    @test logdiag.khat ≈ rawdiag.khat
    shifted, _ = pareto_smooth(log.(positive) .+ 1500; is_log_scale=true)
    @test shifted .- 1500 ≈ logsmoothed atol=1e-10
    integers, _ = pareto_smooth(collect(1:100))
    @test eltype(integers) == Float64
    @test eltype(first(pareto_smooth(Float32.(x)))) == Float32
    for invalid in (ones(100), collect(1.0:20), Float64[], [NaN; vec(x)], [Inf; vec(x)])
        smoothed, d = pareto_smooth(invalid)
        @test isequal(smoothed, invalid)
        @test isnan(d.khat)
    end
    @test_throws ArgumentError pareto_smooth(x; r_eff=0)
    @test_throws ArgumentError pareto_smooth(x; tail=:invalid)
    @test_throws DimensionMismatch pareto_smooth(x; r_eff=ones(2))

    @testset "batches" begin
        values = reshape(x, 100, 2, 2, 5)
        reff = fill(0.8, 2, 5)
        smoothed, d = @inferred pareto_smooth(values; r_eff=reff)
        d2 = pareto_diagnose(values; r_eff=reff)
        for k in propertynames(d)
            @test getproperty(d, k) == getproperty(d2, k)
        end
        for i in CartesianIndices(reff)
            onevar, _ = pareto_smooth(values[:, :, i]; r_eff=reff[i])
            @test smoothed[:, :, i] == onevar
        end
        named = DimArray(
            reshape(x, 100, 2, 10),
            (
                Dimensions.Dim{:draw}(1:100),
                Dimensions.Dim{:chain}(1:2),
                Dimensions.Dim{:param}(Symbol.("x", 1:10)),
            ),
        )
        named_smoothed, named_d = @inferred pareto_smooth(named)
        @test Dimensions.dims(named_smoothed) == Dimensions.dims(named)
        @test Dimensions.dims(named_d.khat) == Dimensions.dims(named, (:param,))
    end
end
