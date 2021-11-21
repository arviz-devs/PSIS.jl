using PSIS
using Test
using Random
using ReferenceTests
using Distributions: Normal, Cauchy, Exponential, logpdf, mean
using LogExpFunctions: logsumexp
using Logging: SimpleLogger, with_logger

@testset "psis/psis!" begin
    @testset "importance sampling tests" begin
        @testset "Exponential($λ) → Exponential(1)" for (λ, klb, kub, rtol) in [
            (0.8, 0, 0.5, 0.02), (0.4, 0.5, 0.7, 0.05), (0.2, 0.7, 1, 0.3)
        ]
            rng = MersenneTwister(42)
            proposal = Exponential(λ)
            target = Exponential(1)
            x = rand(rng, proposal, 10_000)
            logr = logpdf.(target, x) .- logpdf.(proposal, x)
            logr_norm = logsumexp(logr)
            @test sum(exp.(logr .- logr_norm) .* x) ≈ mean(target) rtol = rtol

            logw, k = psis(logr)
            @test klb < k < kub
            logw_norm = logsumexp(logw)
            @test sum(exp.(logw .- logw_norm) .* x) ≈ mean(target) rtol = rtol
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
