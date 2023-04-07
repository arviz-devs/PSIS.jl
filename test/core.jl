using PSIS
using Test
using Random
using ReferenceTests
using Distributions: GeneralizedPareto, Normal, Cauchy, Exponential, TDist, logpdf
using LogExpFunctions: logsumexp, softmax
using Logging: SimpleLogger, with_logger
using AxisArrays: AxisArrays

@testset "PSISResult" begin
    @testset "vector log-weights" begin
        log_weights = randn(500)
        log_weights_norm = logsumexp(log_weights)
        tail_length = 100
        reff = 2.0
        tail_dist = PSIS.GeneralizedPareto(1.0, 1.0, 0.5)
        result = PSISResult(log_weights, log_weights_norm, reff, tail_length, tail_dist)
        @test result isa PSISResult{Float64}
        @test sort(propertynames(result)) == [
            :log_weights,
            :log_weights_norm,
            :nchains,
            :ndraws,
            :nparams,
            :pareto_shape,
            :reff,
            :tail_dist,
            :tail_length,
            :weights,
        ]
        @test result.log_weights == log_weights
        @test result.log_weights_norm == log_weights_norm
        @test result.weights ≈ softmax(log_weights)
        @test result.reff == reff
        @test result.nparams == 1
        @test result.ndraws == 500
        @test result.nchains == 1
        @test result.tail_length == tail_length
        @test result.tail_dist == tail_dist
        @test result.pareto_shape == 0.5
        @test result.ess ≈ ess_is(result)

        @testset "show" begin
            @test sprint(show, "text/plain", result) == """
                PSISResult with 500 draws, 1 chains, and 1 parameters
                Pareto shape (k) diagnostic values:
                                    Count       Min. ESS
                 (-Inf, 0.5]  good  1 (100.0%)  $(floor(Int, result.ess))"""
        end
    end

    @testset "array log-weights" begin
        log_weights = randn(500, 4, 3)
        log_weights_norm = dropdims(logsumexp(log_weights; dims=(1, 2)); dims=(1, 2))
        tail_length = [1600, 1601, 1602]
        reff = [0.8, 0.9, 1.1]
        tail_dist = [
            PSIS.GeneralizedPareto(1.0, 1.0, 0.5),
            PSIS.GeneralizedPareto(1.0, 1.0, 0.6),
            PSIS.GeneralizedPareto(1.0, 1.0, 0.7),
        ]
        result = PSISResult(log_weights, log_weights_norm, reff, tail_length, tail_dist)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.log_weights_norm == log_weights_norm
        @test result.weights ≈ softmax(log_weights; dims=(1, 2))
        @test result.reff == reff
        @test result.nparams == 3
        @test result.ndraws == 500
        @test result.nchains == 4
        @test result.tail_length == tail_length
        @test result.tail_dist == tail_dist
        @test result.pareto_shape == [0.5, 0.6, 0.7]

        @testset "show" begin
            proposal = Normal()
            target = TDist(7)
            rng = MersenneTwister(42)
            x = rand(rng, proposal, 100, 30)
            log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
            reff = [100; ones(29)]
            result = psis(log_ratios, reff)
            @test sprint(show, "text/plain", result) == """
                PSISResult with 100 draws, 1 chains, and 30 parameters
                Pareto shape (k) diagnostic values:
                                        Count       Min. ESS
                 (-Inf, 0.5]  good       2 (6.7%)   98
                  (0.5, 0.7]  okay       6 (20.0%)  92
                    (0.7, 1]  bad        4 (13.3%)  ——
                    (1, Inf)  very bad  17 (56.7%)  ——
                          ——  missing    1 (3.3%)   ——"""
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
            for sz in ((100_000,), (100_000, 5), (100_000, 4, 5))
                dims = length(sz) == 1 ? Colon() : 1:(length(sz) - 1)
                rng = MersenneTwister(42)
                x = rand(rng, proposal, sz)
                logr = logpdf.(target, x) .- logpdf.(proposal, x)

                r = psis(logr)
                @test r isa PSISResult
                logw = r.log_weights
                @test logw isa typeof(logr)

                if length(sz) == 3
                    @test all(r.tail_length .== PSIS.tail_length(1, 400_000))
                else
                    @test all(r.tail_length .== PSIS.tail_length(1, 100_000))
                end

                k = r.pareto_shape
                @test k isa (length(sz) == 1 ? Number : AbstractVector)
                tail_dist = r.tail_dist
                if length(sz) == 1
                    @test tail_dist isa PSIS.GeneralizedPareto
                    @test tail_dist.k == k
                else
                    @test tail_dist isa Vector{<:PSIS.GeneralizedPareto}
                    @test map(d -> d.k, tail_dist) == k
                end

                w = r.weights
                @test all(x -> isapprox(x, k_exp; atol=0.15), k)
                @test all(x -> isapprox(x, x_target; atol=atol), sum(x .* w; dims=dims))
                @test all(
                    x -> isapprox(x, x²_target; atol=atol), sum(x .^ 2 .* w; dims=dims)
                )
            end
        end
    end

    @testset "keywords" begin
        @testset "sorted=true" begin
            x = randn(100)
            perm = sortperm(x)
            @test psis(x).log_weights ==
                invpermute!(psis(x[perm]; sorted=true).log_weights, perm)
            @test psis(x).pareto_shape == psis(x[perm]; sorted=true).pareto_shape
        end
    end

    @testset "warnings" begin
        io = IOBuffer()
        logr = randn(5)
        result = with_logger(SimpleLogger(io)) do
            psis(logr)
        end
        @test result.log_weights == logr
        @test ismissing(result.tail_dist)
        @test ismissing(result.pareto_shape)
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
            @test ismissing(result.tail_dist)
            @test ismissing(result.pareto_shape)
            msg = String(take!(io))
            @test occursin("Warning: Tail contains non-finite values.", msg)
        end

        io = IOBuffer()
        rng = MersenneTwister(42)
        x = rand(rng, Exponential(50), 1_000)
        logr = logpdf.(Exponential(1), x) .- logpdf.(Exponential(50), x)
        result = psis(logr)
        result = with_logger(SimpleLogger(io)) do
            psis(logr)
        end
        @test result.log_weights != logr
        @test result.pareto_shape > 0.7
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 0.73 > 0.7. $(PSIS.BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(PSIS.GeneralizedPareto(0.0, 1.0, 1.1))
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 1.1 > 1. $(PSIS.VERY_BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(PSIS.GeneralizedPareto(0.0, 1.0, 0.8))
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape k = 0.8 > 0.7. $(PSIS.BAD_SHAPE_SUMMARY)", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(PSIS.GeneralizedPareto(0.0, 1.0, 0.69))
        end
        msg = String(take!(io))
        @test isempty(msg)

        tail_dist = [
            missing,
            PSIS.GeneralizedPareto(0, 1, 0.69),
            PSIS.GeneralizedPareto(0, 1, 0.71),
            PSIS.GeneralizedPareto(0, 1, 1.1),
        ]
        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(tail_dist)
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
        rng = MersenneTwister(42)
        proposal = Normal()
        target = Cauchy()
        sz = (5, 1_000, 4)
        x = rand(rng, proposal, sz)
        logr = logpdf.(target, x) .- logpdf.(proposal, x)
        logr = permutedims(logr, (2, 3, 1))
        @testset for r_eff in (0.7, 1.2), improved in (true, false)
            r_effs = fill(r_eff, sz[1])
            result = psis(logr, r_effs; improved=improved)
            logw = result.log_weights
            @test !isapprox(logw, logr)
            basename = "normal_to_cauchy_reff_$(r_eff)"
            if improved
                basename = basename * "_improved"
            end
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

        @testset "AxisArrays" begin
            logr = AxisArrays.AxisArray(
                x,
                AxisArrays.Axis{:iter}(iter_names),
                AxisArrays.Axis{:chain}(chain_names),
                AxisArrays.Axis{:param}(param_names),
            )
            result = psis(logr)
            @test result.log_weights isa AxisArrays.AxisArray
            @test AxisArrays.axes(result.log_weights) == AxisArrays.axes(logr)
            for k in (:pareto_shape, :tail_length, :tail_dist, :reff, :log_weights_norm)
                prop = getproperty(result, k)
                @test prop isa AxisArrays.AxisArray
                @test AxisArrays.axes(prop) == (AxisArrays.axes(logr, 3),)
            end
        end
    end
end
