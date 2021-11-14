using Distributions
using PSIS
using Random
using Test

@testset "Generalized Pareto distribution" begin
    @testset "fit" begin
        @testset for μ in (-1, 5),
            σ in (0.5, 1.0, 2.0),
            ξ in (-1.0, 0.0, 0.3, 1.0, 2.0),
            improved in (true, false)

            d = GeneralizedPareto(μ, σ, ξ)
            rng = MersenneTwister(42)
            x = rand(rng, d, 200_000)
            dhat = fit(
                PSIS.GeneralizedParetoKnownMu(μ), x; min_points=80, improved=improved
            )
            @test dhat.μ == μ
            @test dhat.σ ≈ σ atol = 0.01
            @test dhat.ξ ≈ ξ atol = 0.01
        end
        @testset "nearly uniform" begin
            x = ones(200_000)
            dhat = fit(PSIS.GeneralizedParetoKnownMu(1.0), x; min_points=80)
            @test dhat.μ == 1.0
            @test dhat.σ ≈ eps(0.0)
            @test dhat.ξ ≈ -1
        end
    end
end
