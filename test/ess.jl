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
    result = PSISResult(logw, 0.6, 1.5)
    @test ess_is(result) ≈ ess_is(importance_weights(result); reff=1.5)

    result = PSISResult(logw, 0.71, 1.5)
    @test isnan(ess_is(result))
    @test ess_is(result; bad_shape_nan=false) ≈ ess_is(importance_weights(result); reff=1.5)

    logw = randn(100, 4, 3)
    pareto_shape = [0.69, 0.71, NaN]
    reff = [1.5, 0.8, 1.0]
    result = PSISResult(logw, pareto_shape, reff)
    ess = ess_is(result)
    @test ess isa Vector
    @test length(ess) == 3
    @test ess[1] ≈ ess_is(importance_weights(result); reff=reff)[1]
    @test isnan(ess[2])
    @test isnan(ess[3])
    ess = ess_is(result; bad_shape_nan=false)
    @test ess ≈ ess_is(importance_weights(result); reff=reff)[1:3]
end
