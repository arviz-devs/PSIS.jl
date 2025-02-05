using Distributions
using PSIS
using StableRNGs
using Test

@testset "Generalized Pareto distribution" begin
    @testset "fit_gpd" begin
        @testset for μ in (-1, 5), σ in (0.5, 1.0, 2.0), k in (-1.0, 0.0, 0.3, 1.0, 2.0)
            d = Distributions.GeneralizedPareto(μ, σ, k)
            rng = StableRNG(42)
            x = rand(rng, d, 100_000)
            dhat = PSIS.fit_gpd(x; μ=μ, min_points=80)
            @test dhat.μ == μ
            @test dhat.σ ≈ σ atol = 0.02
            @test dhat.k ≈ k atol = 0.02
            dhat2 = PSIS.fit_gpd(x; prior_adjusted=false, μ=μ, min_points=80)
            @test dhat ≠ dhat2
            @test dhat == PSIS.prior_adjust_shape(dhat2, length(x))
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
