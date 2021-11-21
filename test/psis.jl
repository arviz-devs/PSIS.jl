using PSIS
using Test
using Random
using ReferenceTests
using Distributions: Normal, Cauchy, Exponential, logpdf, mean
using LogExpFunctions: softmax
using Logging: SimpleLogger, with_logger

@testset "psis/psis!" begin
    @testset "importance sampling tests" begin
        target = Exponential(1)
        x_target = 1  # 𝔼[x] with x ~ Exponential(1)
        x²_target = 2  # 𝔼[x²] with x ~ Exponential(1)
        # For θ < 1, the closed-form distribution of importance ratios with ξ = 1 - θ is
        # GeneralizedPareto(θ, θ * ξ, ξ), and the closed-form distribution of tail ratios is
        # GeneralizedPareto(5^ξ * θ, θ * ξ, ξ).
        # For θ < 0.5, the tail distribution has no variance, and estimates with importance
        # weights become unstable
        @testset "Exponential($θ) → Exponential(1)" for (θ, atol) in [
            (0.8, 0.02), (0.55, 0.2), (0.3, 0.6)
        ]
            proposal = Exponential(θ)
            ξ_exp = 1 - θ
            for sz in ((100_000,), (100_000, 4), (5, 100_000, 4))
                dims = length(sz) == 3 ? (2, 3) : Colon()
                r_eff = length(sz) == 3 ? ones(sz[1]) : 1.0
                rng = MersenneTwister(42)
                x = rand(rng, proposal, sz)
                logr = logpdf.(target, x) .- logpdf.(proposal, x)

                logw, k = psis(logr, r_eff)
                w = softmax(logr; dims=dims)
                @test all(≈(ξ_exp; atol=0.1), k)
                @test all(≈(x_target; atol=atol), sum(x .* w; dims=dims))
                @test all(≈(x²_target; atol=atol), sum(x .^ 2 .* w; dims=dims))
            end
        end
    end

    @testset "keywords" begin
        @testset "sorted=true" begin
            x = randn(100)
            perm = sortperm(x)
            @test psis(x, 1.0)[1] == invpermute!(psis(x[perm], 1.0; sorted=true)[1], perm)
            @test psis(x, 1.0)[2] == psis(x[perm], 1.0; sorted=true)[2]
        end

        @testset "normalize=true" begin
            x = randn(100)
            lw1, k1 = psis(x)
            lw2, k2 = psis(x; normalize=true)
            @test k1 == k2
            @test !(lw1 ≈ lw2)
            @test all(abs.(diff(lw1 .- lw2)) .< sqrt(eps()))
            @test sum(exp.(lw2)) ≈ 1
        end
    end

    @testset "warnings" begin
        io = IOBuffer()
        logr = randn(5)
        logw, k = with_logger(SimpleLogger(io)) do
            psis(logr, 1.0)
        end
        @test logw == logr
        @test isinf(k)
        msg = String(take!(io))
        @test occursin(
            "Warning: Insufficient tail draws to fit the generalized Pareto distribution",
            msg,
        )

        io = IOBuffer()
        logr = ones(100)
        logw, k = with_logger(SimpleLogger(io)) do
            psis(logr, 1.0)
        end
        @test logw == logr
        @test isinf(k)
        msg = String(take!(io))
        @test occursin(
            "Warning: Cannot fit the generalized Pareto distribution because all tail values are the same",
            msg,
        )

        io = IOBuffer()
        x = rand(Exponential(100), 1_000)
        logr = logpdf.(Exponential(1), x) .- logpdf.(Exponential(1000), x)
        logw, k = with_logger(SimpleLogger(io)) do
            psis(logr, 1.0)
        end
        @test logw != logr
        @test k > 0.7
        msg = String(take!(io))
        @test occursin(
            "Resulting importance sampling estimates are likely to be unstable", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_k(1.1)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto k=1.1 ≥ 1. Resulting importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples.",
            msg,
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_k(0.8)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto k=0.8 ≥ 0.7. Resulting importance sampling estimates are likely to be unstable.",
            msg,
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_k(0.69)
        end
        msg = String(take!(io))
        @test isempty(msg)
    end

    @testset "test against reference values" begin
        rng = MersenneTwister(42)
        proposal = Normal()
        target = Cauchy()
        x = rand(rng, proposal, 1_000)
        logr = logpdf.(target, x) .- logpdf.(proposal, x)
        expected_khats = Dict(
            (0.7, false) => 0.87563321,
            (1.2, false) => 0.99029843,
            (0.7, true) => 0.88650519,
            (1.2, true) => 1.00664484,
        )
        @testset for r_eff in (0.7, 1.2), improved in (true, false)
            logw, k = psis(logr, r_eff; improved=improved)
            @test !isapprox(logw, logr)
            basename = "normal_to_cauchy_reff_$(r_eff)"
            if improved
                basename = basename * "_improved"
            end
            @test_reference(
                "references/$basename.jld2",
                Dict("data" => logw),
                by = (ref, x) -> isapprox(ref["data"], x["data"]; rtol=1e-6),
            )
            k_ref = expected_khats[(r_eff, improved)]
            @test k ≈ k_ref
        end
    end
end
