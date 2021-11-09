using PSIS
using Test
using RCall
using Random
using Distributions: Exponential, GeneralizedPareto, logpdf, mean
using LogExpFunctions: logsumexp
using Logging: SimpleLogger, with_logger

function has_loo()
    R"has_loo <- require('loo')"
    return @rget has_loo
end

function psis_loo(logr, r_eff)
    R"""
    res <- psis($logr, r_eff=$r_eff)
    logw <- res$log_weights
    k <- res$diagnostics$pareto_k
    """
    return vec(@rget(logw)), @rget(k)
end

@testset "psis/psis!" begin
    @testset "basic" begin
        # std normal proposal and target
        x = -randn(1000) .^ 2 ./ 2
        res = @inferred PSISResult{Float64} psis(x, 0.7)
        @test res isa PSISResult
        logsumw = logsumexp(res.log_weights)
        @test sum(res.weights) ≈ 1
        @test res.weights ≈ exp.(res.log_weights .- logsumw)
        @test res.ndraws == length(res.log_weights)
        @test res.pareto_k < 0.5
        tail_length = PSIS.tail_length(0.7, 1_000)
        @test res.tail_length == tail_length
        tail_dist = res.tail_dist
        @test tail_dist isa GeneralizedPareto
        @test tail_dist.ξ == res.pareto_k
    end

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

            res = psis(logr, 1.0)
            logsumw = logsumexp(res.log_weights)
            @test klb < res.pareto_k < kub
            @test sum(res.weights .* x) ≈ mean(target) rtol = rtol
        end
    end

    @testset "keywords" begin
        @testset "sorted=true" begin
            x = randn(100)
            perm = sortperm(x)
            res = psis(x, 1.0)
            res_perm = psis(x[perm], 1.0; sorted=true)
            @test res.log_weights ≈ invpermute!(copy(res_perm.log_weights), perm)
            @test res.weights ≈ invpermute!(copy(res_perm.weights), perm)
            @test res.pareto_k ≈ res_perm.pareto_k
        end
    end

    @testset "warnings" begin
        io = IOBuffer()
        logr = randn(5)
        res = with_logger(SimpleLogger(io)) do
            psis(logr, 1.0)
        end
        @test res.log_weights == logr
        @test res.pareto_k isa Missing
        @test res.tail_dist isa Missing
        msg = String(take!(io))
        @test occursin(
            "Warning: Insufficient tail draws to fit the generalized Pareto distribution",
            msg,
        )

        io = IOBuffer()
        x = rand(Exponential(100), 1_000)
        logr = logpdf.(Exponential(1), x) .- logpdf.(Exponential(1000), x)
        res = with_logger(SimpleLogger(io)) do
            psis(logr, 1.0)
        end
        @test res.log_weights != logr
        @test res.pareto_k > 0.7
        msg = String(take!(io))
        @test occursin(
            "Resulting importance sampling estimates are likely to be unstable", msg
        )

        io = IOBuffer()
        dist = GeneralizedPareto(0.3, 1.0, 1.1)
        with_logger(SimpleLogger(io)) do
            PSIS.check_tail_dist(dist)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto k=1.1 ≥ 1. Resulting importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples.",
            msg,
        )

        io = IOBuffer()
        dist = GeneralizedPareto(0.3, 1.0, 0.8)
        with_logger(SimpleLogger(io)) do
            PSIS.check_tail_dist(dist)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto k=0.8 ≥ 0.7. Resulting importance sampling estimates are likely to be unstable.",
            msg,
        )

        io = IOBuffer()
        dist = GeneralizedPareto(0.3, 1.0, 0.69)
        with_logger(SimpleLogger(io)) do
            PSIS.check_tail_dist(dist)
        end
        msg = String(take!(io))
        @test isempty(msg)
    end

    has_loo() && @testset "consistent with loo" begin
        n = 10_000
        @testset for r_eff in [0.1, 0.5, 0.9, 1.0, 1.2]
            logr = randn(n)
            res = psis(logr, r_eff)
            @test !isapprox(res.log_weights, logr)
            logw_loo, k_loo = psis_loo(logr, r_eff)
            @test res.log_weights ≈ logw_loo
            @test res.pareto_k ≈ k_loo
        end
    end
end
