using PSIS
using Random
using Test

@testset "Generalized Pareto distribution" begin
    @testset "consistency of quantile and fit" begin
        # check that when uniform draws from the quantiles are used to fit the parameters,
        # the original parameters are approximately recovered
        Random.seed!(42)
        u = rand(200_000)
        @testset for σ in (0.5, 1.0, 2.0), k in (-1.0, 0.0, 0.3, 1.0, 2.0)
            d = PSIS.GeneralizedPareto(σ, k)
            x = PSIS.quantile.(Ref(d), u)
            dhat = PSIS.fit(PSIS.GeneralizedPareto, x; adjust_prior=false, min_points=80)
            @test dhat.σ ≈ σ atol = 0.01
            @test dhat.k ≈ k atol = 0.01
        end
    end
end
