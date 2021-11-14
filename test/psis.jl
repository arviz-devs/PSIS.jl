using PSIS
using Test
using RCall
using Distributions: Exponential, logpdf, mean
using LogExpFunctions: logsumexp
using Logging: SimpleLogger, with_logger

function has_loo()
    R"has_loo <- require('loo')"
    return @rget has_loo
end

function psis_loo(logr, r_eff=1.0)
    R"""
    res <- psis($logr, r_eff=$r_eff)
    logw <- res$log_weights
    k <- res$diagnostics$pareto_k
    """
    return vec(@rget(logw)), @rget(k)
end

@testset "psis/psis!" begin
    @testset "importance sampling tests" begin
        @testset "Exponential($λ) → Exponential(1)" for (λ, klb, kub, rtol) in [
            (0.8, 0, 0.5, 0.02), (0.4, 0.5, 0.7, 0.05), (0.2, 0.7, 1, 0.3)
        ]
            Random.seed!(42)
            proposal = Exponential(λ)
            target = Exponential(1)
            x = rand(proposal, 10_000)
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
            @test psis(x)[1] == invpermute!(psis(x[perm]; sorted=true)[1], perm)
            @test psis(x)[2] == psis(x[perm]; sorted=true)[2]
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
            psis(logr)
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
            psis(logr)
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
            psis(logr)
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

    has_loo() && @testset "consistent with loo" begin
        n = 10_000
        @testset for r_eff in [0.1, 0.5, 0.9, 1.0, 1.2]
            logr = randn(n)
            logw, k = psis(logr, r_eff; improved=false)
            @test !isapprox(logw, logr)
            logw_loo, k_loo = psis_loo(logr, r_eff)
            @test logw ≈ logw_loo
            @test k ≈ k_loo
        end
    end
end
