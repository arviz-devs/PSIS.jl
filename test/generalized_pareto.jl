using Distributions
using PSIS
using Random
using Test

@testset "Generalized Pareto distribution" begin
    @testset "fit_gpd" begin
        @testset for μ in (-1, 5), σ in (0.5, 1.0, 2.0), k in (-1.0, 0.0, 0.3, 1.0, 2.0)
            d = Distributions.GeneralizedPareto(μ, σ, k)
            rng = MersenneTwister(42)
            x = rand(rng, d, 200_000)
            dhat = PSIS.fit_gpd(x; μ=μ, min_points=80)
            @test dhat.μ == μ
            @test dhat.σ ≈ σ atol = 0.01
            @test dhat.k ≈ k atol = 0.01
        end
        @testset "nearly uniform" begin
            x = ones(200_000)
            dhat = PSIS.fit_gpd(x; μ=1.0, min_points=80)
            @test dhat.μ == 1.0
            @test dhat.σ == 0.0
            @test dhat.k == -1
        end
    end
end
