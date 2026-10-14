using Distributions
using LogExpFunctions
using PSIS
using Test

@testset "effective sample size" begin
    logw = fill(randn(), 100)
    @test PSIS.ess_is(logw, 1) ≈ 100
    r_eff = rand()
    @test PSIS.ess_is(logw, r_eff) ≈ 100 * r_eff

    logw = fill(-Inf, 100)
    i = rand(1:100)
    logw[i] = 0
    @test PSIS.ess_is(logw, 1) ≈ 1
    @test PSIS.ess_is(logw, r_eff) ≈ r_eff

    logw = randn(100, 4, 3)
    w = softmax(logw; dims=(1, 2))
    dims = (1, 2)
    @test PSIS.ess_is(logw, r_eff) ≈ r_eff ./ dropdims(sum(abs2, w; dims); dims)
    r_eff = rand(3)
    @test PSIS.ess_is(logw, r_eff) ≈ r_eff ./ dropdims(sum(abs2, w; dims); dims)
end
