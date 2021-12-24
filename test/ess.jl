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
    logw_norm = logsumexp(logw)
    result = PSISResult(logw, logw_norm, 1.5, 20, GeneralizedPareto(0.0, 1.0, 0.6))
    @test ess_is(result) ≈ ess_is(exp.(logw .- logw_norm); reff=1.5)

    result = PSISResult(logw, logw_norm, 1.5, 20, GeneralizedPareto(0.0, 1.0, 0.71))
    @test ismissing(ess_is(result))
    @test ess_is(result; bad_shape_missing=false) ≈
        ess_is(exp.(logw .- logw_norm); reff=1.5)

    logw = randn(3, 100, 4)
    logw_norm = dropdims(logsumexp(logw; dims=(2, 3)); dims=(2, 3))
    tail_dists = [
        GeneralizedPareto(0.0, 1.0, 0.69), GeneralizedPareto(0.0, 1.0, 0.71), missing
    ]
    reff = [1.5, 0.8, 1.0]
    result = PSISResult(logw, logw_norm, reff, [20, 20, 20], tail_dists)
    ess = ess_is(result)
    @test ess isa Vector
    @test length(ess) == 3
    @test ess[1] ≈ ess_is(result.weights; reff=reff)[1]
    @test ismissing(ess[2])
    @test ismissing(ess[3])
    ess = ess_is(result; bad_shape_missing=false)
    @test ess[1:2] ≈ ess_is(result.weights; reff=reff)[1:2]
    @test ismissing(ess[3])
end
