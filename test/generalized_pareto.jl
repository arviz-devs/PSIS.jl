using Distributions
using PSIS
using Random
using Test

@testset "Generalized Pareto distribution" begin
    @testset "fit" begin
        Random.seed!(42)
        @testset for μ in (-1, 5), σ in (0.5, 1.0, 2.0), ξ in (-1.0, 0.0, 0.3, 1.0, 2.0)
            d = GeneralizedPareto(μ, σ, ξ)
            x = rand(d, 200_000)
            dhat = fit(PSIS.GeneralizedParetoKnownMu(μ), x; min_points=80)
            @test dhat.σ ≈ σ atol = 0.01
            @test dhat.ξ ≈ ξ atol = 0.01
        end
    end
end
