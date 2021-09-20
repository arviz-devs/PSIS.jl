using PSIS
using Test
using RCall

function install_loo()
    R"""
    install.packages("loo",
                     dependencies=TRUE,
                     install.packages.check.source="no",
                     repos=c("https://cloud.r-project.org"))
    library(loo)
    """
end

function psis_loo(logr, r_eff=1.0)
    R"""
    res <- psis($logr, r_eff=$r_eff)
    logw <- res$log_weights
    k <- res$diagnostics$pareto_k
    """
    return vec(@rget(logw)), @rget(k)
end

install_loo()

@testset "psis/psis!" begin
    @testset "consistent with loo" begin
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
