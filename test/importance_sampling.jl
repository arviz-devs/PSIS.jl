using PSIS
using Test
using ReferenceTests
using StableRNGs
using Distributions: GeneralizedPareto, Normal, Cauchy, Exponential, TDist, logpdf
using LogExpFunctions: logsumexp, softmax
using DimensionalData: Dimensions, DimArray

@testset "PSISResult" begin
    @testset "vector log-weights" begin
        log_weights = randn(500)
        ess_is = 300.0
        pareto_khat = 0.5
        result = PSISResult(log_weights, pareto_khat, ess_is)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.pareto_khat == pareto_khat
        @test result.ess_is == ess_is

        @testset "show" begin
            @test sprint(show, "text/plain", result) == """
                PSISResult with 500 draws, 1 chains, and 1 parameters
                Pareto k-hat diagnostic summary:
                                        Count       Min. ESS
                 (-Inf, 0.629]  (good)  1 (100.0%)  $(floor(Int, ess_is))"""
        end
    end

    @testset "array log-weights" begin
        log_weights = randn(500, 4, 3)
        r_eff = [0.8, 0.9, 1.1]
        ess_is = [100.0, 101.0, 102.0]
        pareto_khats = [0.5, 0.6, 0.7]
        result = PSISResult(log_weights, pareto_khats, ess_is)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.pareto_khat == pareto_khats
        @test result.ess_is == ess_is
        @testset "show" begin
            proposal = Normal()
            target = TDist(7)
            rng = StableRNG(42)
            x = rand(rng, proposal, 100, 1, 30)
            log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
            r_eff = [100; ones(29)]
            result = psis(log_ratios, r_eff; warn=false)
            @test sprint(show, "text/plain", result) == """
                PSISResult with 100 draws, 1 chains, and 30 parameters
                Pareto k-hat diagnostic summary:
                                       Count       Min. ESS
                 (0.5, 1]  (bad)        4 (13.3%)  ——
                 (1, Inf)  (very bad)  25 (83.3%)  ——
                       ——  (failed)     1 (3.3%)   ——"""
        end
    end
end

@testset "psis" begin
    @testset "importance sampling tests" begin
        target = Exponential(1)
        x_target = 1  # 𝔼[x] with x ~ Exponential(1)
        x²_target = 2  # 𝔼[x²] with x ~ Exponential(1)
        # For θ < 1, the closed-form distribution of importance ratios with k = 1 - θ is
        # GeneralizedPareto(θ, θ * k, k), and the closed-form distribution of tail ratios is
        # GeneralizedPareto(5^k * θ, θ * k, k).
        # For θ < 0.5, the tail distribution has no variance, and estimates with importance
        # weights become unstable
        @testset "Exponential($θ) → Exponential(1)" for (θ, atol) in [
            (0.8, 0.05), (0.55, 0.2), (0.3, 0.7)
        ]
            proposal = Exponential(θ)
            k_exp = 1 - θ
            for sz in ((100_000,), (100_000, 4), (100_000, 4, 5))
                dims = length(sz) < 3 ? Colon() : 1:(length(sz) - 1)
                rng = StableRNG(42)
                x = rand(rng, proposal, sz)
                logr = logpdf.(target, x) .- logpdf.(proposal, x)

                r = @inferred psis(logr; warn=false)
                @test r isa PSISResult
                logw = r.log_weights
                @test logw isa typeof(logr)

                k = r.pareto_khat
                @test k isa (length(sz) < 3 ? Number : AbstractVector)
                @test r.pareto_khat == k

                w = softmax(logw; dims=1:min(2, length(sz)))
                @test all(x -> isapprox(x, k_exp; atol=0.15), k)
                @test all(x -> isapprox(x, x_target; atol=atol), sum(x .* w; dims=dims))
                @test all(
                    x -> isapprox(x, x²_target; atol=atol), sum(x .^ 2 .* w; dims=dims)
                )
            end
        end
    end

    @testset "r_eff combinations" begin
        r_effs_uniform = [rand(), fill(rand()), [rand()]]
        x = randn(1000)
        for r in r_effs_uniform
            psis(x, r)
        end
        @test_throws DimensionMismatch psis(x, rand(2))

        x = randn(1000, 4)
        for r in r_effs_uniform
            psis(x, r)
        end
        @test_throws DimensionMismatch psis(x, rand(2))

        x = randn(1000, 4, 2)
        for r in r_effs_uniform
            psis(x, r)
        end
        psis(x, rand(2))
        @test_throws DimensionMismatch psis(x, rand(3))

        x = randn(1000, 4, 2, 3)
        for r in r_effs_uniform
            psis(x, r)
        end
        psis(x, rand(2, 3))
        @test_throws DimensionMismatch psis(x, rand(3))
    end

    @testset "invalid inputs and failed fits" begin
        for sz in (100, (100, 4, 3)), rbad in (-1, 0, NaN, Inf)
            @test_throws ArgumentError psis(randn(sz), rbad)
        end
        for logr in (
            randn(5),
            [NaN; randn(100)],
            [Inf; randn(100)],
            fill(-Inf, 100),
            vcat(ones(50), fill(-Inf, 435)),
        )
            result = @test_logs (:warn, r"could not be fit") psis(logr)
            @test isequal(result.log_weights, logr)
            @test isnan(result.pareto_khat)
            @test isnan(result.ess_is)
        end
    end

    @testset "test against reference values" begin
        rng = StableRNG(42)
        proposal = Normal()
        target = Cauchy()
        sz = (5, 1_000, 4)
        x = rand(rng, proposal, sz)
        logr = logpdf.(target, x) .- logpdf.(proposal, x)
        logr = permutedims(logr, (2, 3, 1))
        @testset for r_eff in (0.7, 1.2)
            r_effs = fill(r_eff, sz[1])
            # Freeze the historical ceiling-based tail count for these fixtures.
            # The v9 floor-based default is tested separately in test/tails.jl.
            n = size(logr, 1) * size(logr, 2)
            m = ceil(Int, min(n / 5, 3sqrt(n / r_eff)))
            result = @inferred psis(logr, r_effs; tail_length=m, warn=false)
            logw = result.log_weights
            @test !isapprox(logw, logr)
            basename = "normal_to_cauchy_r_eff_$(r_eff)"
            @test_reference(
                "references/$basename.jld2",
                Dict(
                    "log_weights" => logw,
                    "pareto_khat" => result.pareto_khat,
                    "ess_is" => PSIS.ess_is(logw, r_effs),
                ),
                by =
                    (ref, x) ->
                        isapprox(ref["log_weights"], x["log_weights"]; rtol=1e-6) &&
                        isapprox(ref["pareto_khat"], x["pareto_khat"]; rtol=1e-6) &&
                        isapprox(ref["ess_is"], x["ess_is"]; rtol=1e-6)
            )
        end
    end

    # https://github.com/arviz-devs/PSIS.jl/issues/27
    @testset "no failure for very low log-weights" begin
        psis(rand(1000) .- 1500)
    end

    @testset "compatibility with arrays with named axes/dims" begin
        param_names = [Symbol("x[$i]") for i in 1:10]
        iter_names = 101:200
        chain_names = 1:4
        x = randn(length(iter_names), length(chain_names), length(param_names))

        @testset "DimensionalData" begin
            logr = DimArray(
                x,
                (
                    Dimensions.Dim{:iter}(iter_names),
                    Dimensions.Dim{:chain}(chain_names),
                    Dimensions.Dim{:param}(param_names),
                ),
            )
            result = @inferred psis(logr; warn=false)
            @test result.log_weights isa DimArray
            @test Dimensions.dims(result.log_weights) == Dimensions.dims(logr)
            @testset for k in (:pareto_khat, :ess_is)
                prop = getfield(result, k)
                @test prop isa DimArray
                @test Dimensions.dims(prop) == Dimensions.dims(logr, (:param,))
            end
        end
    end
end
