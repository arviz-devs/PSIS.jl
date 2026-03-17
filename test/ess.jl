using Distributions
using LogExpFunctions
using PSIS
using Test

@testset "effective sample size" begin
    w = fill(0.01, 100)
    r_eff = rand()
    @test ess_is(w) ≈ 100
    @test ess_is(w; r_eff) ≈ 100 .* r_eff

    w = zeros(100)
    i = rand(1:100)
    w[i] = 1
    @test ess_is(w) ≈ 1
    @test ess_is(w; r_eff) ≈ 1 .* r_eff

    logw = randn(100)
    result = PSISResult(logw, 1.5, 20, PSIS.GeneralizedPareto(0.0, 1.0, 0.6))
    weights = PSIS.importance_weights(logw)
    @test ess_is(result) ≈ ess_is(weights; r_eff=1.5)

    result = PSISResult(logw, 1.5, 20, PSIS.GeneralizedPareto(0.0, 1.0, 0.71))
    @test isnan(ess_is(result))
    weights = PSIS.importance_weights(logw)
    @test ess_is(result; bad_shape_nan=false) ≈ ess_is(weights; r_eff=1.5)

    logw = randn(100, 4, 3)
    tail_dists = [
        PSIS.GeneralizedPareto(0.0, 1.0, 0.69),
        PSIS.GeneralizedPareto(0.0, 1.0, 0.71),
        PSIS.GeneralizedPareto(0.0, NaN, NaN),
    ]
    r_eff = [1.5, 0.8, 1.0]
    result = PSISResult(logw, r_eff, [20, 20, 20], tail_dists)
    ess = ess_is(result)
    @test ess isa Vector
    @test length(ess) == 3
    weights = PSIS.importance_weights(logw)
    @test ess[1] ≈ ess_is(weights; r_eff)[1]
    @test isnan(ess[2])
    @test isnan(ess[3])
    ess = ess_is(result; bad_shape_nan=false)
    weights = PSIS.importance_weights(logw)
    @test ess ≈ ess_is(weights; r_eff)[1:3]
end
