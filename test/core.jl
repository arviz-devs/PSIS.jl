using PSIS
using Test
using ReferenceTests
using StableRNGs
using Distributions: GeneralizedPareto, Normal, Cauchy, Exponential, TDist, logpdf
using LogExpFunctions: logsumexp, softmax
using Logging: SimpleLogger, with_logger
using DimensionalData: Dimensions, DimArray

@testset "PSISResult" begin
    @testset "vector log-weights" begin
        log_weights = randn(500)
        log_weights_norm = logsumexp(log_weights)
        reff = 2.0
        pareto_shape = 0.5
        result = PSISResult(log_weights, pareto_shape, reff)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.reff == reff
        @test result.pareto_shape == pareto_shape
        ess = PSIS.ess_is(result)

        @testset "show" begin
            @test sprint(show, "text/plain", result) == """
                PSISResult with 500 draws, 1 chains, and 1 parameters
                Pareto shape (k) diagnostic values:
                                    Count       Min. ESS
                 (-Inf, 0.5]  good  1 (100.0%)  $(floor(Int, ess))"""
        end
    end

    @testset "array log-weights" begin
        log_weights = randn(500, 4, 3)
        reff = [0.8, 0.9, 1.1]
        pareto_shape = [0.5, 0.6, 0.7]
        result = PSISResult(log_weights, pareto_shape, reff)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.pareto_shape == pareto_shape
        @test result.reff == reff

        @testset "show" begin
            proposal = Normal()
            target = TDist(7)
            rng = StableRNG(42)
            x = rand(rng, proposal, 100, 1, 30)
            log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
            reff = [100; ones(29)]
            result = psis(log_ratios, reff)
            @test sprint(show, "text/plain", result) == """
                PSISResult with 100 draws, 1 chains, and 30 parameters
                Pareto shape (k) diagnostic values:
                                       Count       Min. ESS
                 (0.5, 0.7]  okay       2 (6.7%)   99
                   (0.7, 1]  bad        2 (6.7%)   ——
                   (1, Inf)  very bad  25 (83.3%)  ——
                         ——  failed     1 (3.3%)   ——"""
        end
    end
end

@testset "psis/psis!" begin
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

                r = @inferred psis(logr)
                @test r isa PSISResult
                logw = r.log_weights
                @test logw isa typeof(logr)

                k = r.pareto_shape
                @test k isa (length(sz) < 3 ? Number : AbstractVector)

                w = importance_weights(r)
                @test all(x -> isapprox(x, k_exp; atol=0.15), k)
                @test all(x -> isapprox(x, x_target; atol=atol), sum(x .* w; dims=dims))
                @test all(
                    x -> isapprox(x, x²_target; atol=atol), sum(x .^ 2 .* w; dims=dims)
                )
            end
        end
    end

    @testset "reff combinations" begin
        reffs_uniform = [rand(), fill(rand()), [rand()]]
        x = randn(1000)
        for r in reffs_uniform
            psis(x, r)
        end
        @test_throws DimensionMismatch psis(x, rand(2))

        x = randn(1000, 4)
        for r in reffs_uniform
            psis(x, r)
        end
        @test_throws DimensionMismatch psis(x, rand(2))

        x = randn(1000, 4, 2)
        for r in reffs_uniform
            psis(x, r)
        end
        psis(x, rand(2))
        @test_throws DimensionMismatch psis(x, rand(3))

        x = randn(1000, 4, 2, 3)
        for r in reffs_uniform
            psis(x, r)
        end
        psis(x, rand(2, 3))
        @test_throws DimensionMismatch psis(x, rand(3))
    end

    @testset "warnings" begin
        io = IOBuffer()
        @testset for sz in (100, (100, 4, 3)), rbad in (-1, 0, NaN)
            logr = randn(sz)
            result = with_logger(SimpleLogger(io)) do
                psis(logr, rbad)
            end
            msg = String(take!(io))
            @test occursin("All values of `reff` should be finite, but some are not.", msg)
        end

        io = IOBuffer()
        logr = randn(5)
        result = with_logger(SimpleLogger(io)) do
            psis(logr)
        end
        @test result.log_weights == logr
        msg = String(take!(io))
        @test occursin(
            "Warning: 1 tail draws is insufficient to fit the generalized Pareto distribution.",
            msg,
        )

        skipnan(x) = filter(!isnan, x)
        io = IOBuffer()
        for logr in [
            [NaN; randn(100)],
            [Inf; randn(100)],
            fill(-Inf, 100),
            vcat(ones(50), fill(-Inf, 435)),
        ]
            result = with_logger(SimpleLogger(io)) do
                psis(logr)
            end
            @test skipnan(result.log_weights) == skipnan(logr)
            @test isnan(result.pareto_shape)
            msg = String(take!(io))
            @test occursin("Warning: Tail contains non-finite values.", msg)
        end

        io = IOBuffer()
        rng = StableRNG(83)
        x = rand(rng, Exponential(50), 1_000)
        logr = logpdf.(Exponential(1), x) .- logpdf.(Exponential(50), x)
        result = with_logger(SimpleLogger(io)) do
            psis(logr)
        end
        @test result.log_weights != logr
        @test result.pareto_shape > 0.7
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 0.72 > 0.7. $(PSIS.BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(1.1)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 1.1 > 1. $(PSIS.VERY_BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(0.8)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 0.8 > 0.7. $(PSIS.BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(0.69)
        end
        msg = String(take!(io))
        @test isempty(msg)

        pareto_shape = [NaN, 0.69, 0.71, 1.1]
        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(pareto_shape)
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. $(PSIS.BAD_SHAPE_SUMMARY)",
            msg,
        )
        @test occursin(
            "Warning: 1 parameters had Pareto shape values k > 1. $(PSIS.VERY_BAD_SHAPE_SUMMARY)",
            msg,
        )
        @test occursin(
            "Warning: For 1 parameters, the generalized Pareto distribution could not be fit to the tail draws.",
            msg,
        )
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
            result = @inferred psis(logr, r_effs)
            logw = result.log_weights
            @test !isapprox(logw, logr)
            basename = "normal_to_cauchy_reff_$(r_eff)"
            @test_reference(
                "references/$basename.jld2",
                Dict("log_weights" => logw, "pareto_shape" => result.pareto_shape),
                by =
                    (ref, x) ->
                        isapprox(ref["log_weights"], x["log_weights"]; rtol=1e-6) &&
                        isapprox(ref["pareto_shape"], x["pareto_shape"]; rtol=1e-6),
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
            result = @inferred psis(logr)
            @test result.log_weights isa DimArray
            @test Dimensions.dims(result.log_weights) == Dimensions.dims(logr)
            for k in (:pareto_shape, :reff)
                prop = getproperty(result, k)
                @test prop isa DimArray
                @test Dimensions.dims(prop) == Dimensions.dims(logr, (:param,))
            end
        end
    end
end
