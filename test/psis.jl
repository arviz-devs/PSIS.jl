using PSIS
using Test
using RCall

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
    has_loo() && @testset "consistent with loo" begin
        n = 10_000
        @testset for r_eff in [0.1, 0.5, 0.9, 1.0, 1.2]
            logr = randn(n)
            logw, k = psis(logr, r_eff)
            @test !isapprox(logw, logr)
            logw_loo, k_loo = psis_loo(logr, r_eff)
            @test logw ≈ logw_loo
            @test k ≈ k_loo
        end
    end
end
