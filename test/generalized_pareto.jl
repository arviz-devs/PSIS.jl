using Distributions
using PSIS
using Random
using StatsBase
using Test

@testset "Generalized Pareto distribution" begin
    @testset "consistency of quantile and fit" begin
        # check that when uniform draws from the quantiles are used to fit the parameters,
        # the original parameters are approximately recovered
        Random.seed!(42)
        u = rand(200_000)
        @testset for μ in (0.0, -1.0, 2.0),
            σ in (0.5, 1.0, 2.0),
            ξ in (-1.0, 0.0, 0.3, 1.0, 2.0)

            d = GeneralizedPareto(μ, σ, ξ)
            x = quantile.(Ref(d), u)
            dic = PSIS.GeneralizedParetoKnownMu(μ)
            method = PSIS.EmpiricalBayesEstimate()
            dhat = StatsBase.fit!(dic, x, method; adjust_prior=false, min_points=80)
            @test dhat.μ == μ
            @test dhat.σ ≈ σ atol = 0.01
            @test dhat.ξ ≈ ξ atol = 0.01
        end
    end
    @testset "" begin
        d = GeneralizedPareto(1.1, 1.0, -Inf)
        x = rand(d, 1_000)
        dic = PSIS.GeneralizedParetoKnownMu(1.1)
        method = PSIS.EmpiricalBayesEstimate()
        dhat = StatsBase.fit!(dic, x, method)
        @test dhat.μ == 1.1
        @test dhat.σ == 1.0
        @test dhat.ξ == -Inf
    end
end
