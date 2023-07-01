using Distributions
using LogExpFunctions
using PSIS
using Test

@testset "effective sample size" begin
    w = fill(0.01, 100)
    reff = rand()
    @test ess_is(w) ≈ 100
    @test ess_is(w; reff=reff) ≈ 100 .* reff

    w = zeros(100)
    i = rand(1:100)
    w[i] = 1
    @test ess_is(w) ≈ 1
    @test ess_is(w; reff=reff) ≈ 1 .* reff

    logw = randn(100)
    result = PSISResult(logw, 1.5, 20, PSIS.GeneralizedPareto(0.0, 1.0, 0.6), false)
    @test ess_is(result) ≈ ess_is(result.weights; reff=1.5)

    result = PSISResult(logw, 1.5, 20, PSIS.GeneralizedPareto(0.0, 1.0, 0.71), false)
    @test isnan(ess_is(result))
    @test ess_is(result; bad_shape_nan=false) ≈ ess_is(result.weights; reff=1.5)

    logw = randn(100, 4, 3)
    tail_dists = [
        PSIS.GeneralizedPareto(0.0, 1.0, 0.69),
        PSIS.GeneralizedPareto(0.0, 1.0, 0.71),
        PSIS.GeneralizedPareto(0.0, NaN, NaN),
    ]
    reff = [1.5, 0.8, 1.0]
    result = PSISResult(logw, reff, [20, 20, 20], tail_dists, false)
    ess = ess_is(result)
    @test ess isa Vector
    @test length(ess) == 3
    @test ess[1] ≈ ess_is(result.weights; reff=reff)[1]
    @test isnan(ess[2])
    @test isnan(ess[3])
    ess = ess_is(result; bad_shape_nan=false)
    @test ess ≈ ess_is(result.weights; reff=reff)[1:3]
end
