using PSIS
using Test
using Random
using ReferenceTests
using Distributions: GeneralizedPareto, Normal, Cauchy, Exponential, logpdf, mean, shape
using LogExpFunctions: logsumexp, softmax
using Logging: SimpleLogger, with_logger
using AxisArrays: AxisArrays

@testset "PSISResult" begin
    @testset "vector log-weights" begin
        log_weights = randn(500)
        log_weights_norm = logsumexp(log_weights)
        tail_length = 100
        reff = 2.0
        tail_dist = GeneralizedPareto(1.0, 1.0, 0.5)
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

        @testset "show" begin
            @test sprint(show, "text/plain", result) ==
                "$(typeof(result)):\n    pareto_shape: 0.5"
        end
    end

    @testset "array log-weights" begin
        log_weights = randn(3, 500, 4)
        log_weights_norm = dropdims(logsumexp(log_weights; dims=(2, 3)); dims=(2, 3))
        tail_length = [1600, 1601, 1602]
        reff = [0.8, 0.9, 1.1]
        tail_dist = [
            GeneralizedPareto(1.0, 1.0, 0.5),
            GeneralizedPareto(1.0, 1.0, 0.6),
            GeneralizedPareto(1.0, 1.0, 0.7),
        ]
        result = PSISResult(log_weights, log_weights_norm, reff, tail_length, tail_dist)
        @test result isa PSISResult{Float64}
        @test result.log_weights == log_weights
        @test result.log_weights_norm == log_weights_norm
        @test result.weights ≈ softmax(log_weights; dims=(2, 3))
        @test result.reff == reff
        @test result.nparams == 3
        @test result.ndraws == 500
        @test result.nchains == 4
        @test result.tail_length == tail_length
        @test result.tail_dist == tail_dist
        @test result.pareto_shape == [0.5, 0.6, 0.7]

        @testset "show" begin
            @test sprint(show, "text/plain", result) ==
                "$(typeof(result)):\n    pareto_shape: [0.5, 0.6, 0.7]"
        end
    end
end

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
            (0.8, 0.05), (0.55, 0.2), (0.3, 0.7)
        ]
            proposal = Exponential(θ)
            ξ_exp = 1 - θ
            for sz in ((100_000,), (5, 100_000), (5, 100_000, 4))
                dims = length(sz) == 1 ? Colon() : 2:length(sz)
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

                ξ = r.pareto_shape
                @test ξ isa (length(sz) == 1 ? Number : AbstractVector)
                tail_dist = r.tail_dist
                if length(sz) == 1
                    @test tail_dist isa GeneralizedPareto
                    @test shape(tail_dist) == ξ
                else
                    @test tail_dist isa Vector{<:GeneralizedPareto}
                    @test map(shape, tail_dist) == ξ
                end

                w = r.weights
                @test all(x -> isapprox(x, ξ_exp; atol=0.15), ξ)
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
            "Warning: Insufficient tail draws to fit the generalized Pareto distribution",
            msg,
        )

        io = IOBuffer()
        x = rand(Exponential(100), 1_000)
        logr = logpdf.(Exponential(1), x) .- logpdf.(Exponential(1000), x)
        result = with_logger(SimpleLogger(io)) do
            psis(logr)
        end
        @test result.log_weights != logr
        @test result.pareto_shape > 0.7
        msg = String(take!(io))
        @test occursin(
            "Resulting importance sampling estimates are likely to be unstable", msg
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(GeneralizedPareto(0.0, 1.0, 1.1))
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape=1.1 ≥ 1. Resulting importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples.",
            msg,
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(GeneralizedPareto(0.0, 1.0, 0.8))
        end
        msg = String(take!(io))
        @test occursin(
            "Warning: Pareto shape=0.8 ≥ 0.7. Resulting importance sampling estimates are likely to be unstable.",
            msg,
        )

        io = IOBuffer()
        with_logger(SimpleLogger(io)) do
            PSIS.check_pareto_shape(GeneralizedPareto(0.0, 1.0, 0.69))
        end
        msg = String(take!(io))
        @test isempty(msg)
    end

    @testset "test against reference values" begin
        rng = MersenneTwister(42)
        proposal = Normal()
        target = Cauchy()
        sz = (5, 1_000, 4)
        x = rand(rng, proposal, sz)
        logr = logpdf.(target, x) .- logpdf.(proposal, x)
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

    @testset "compatibility with arrays with named axes/dims" begin
        param_names = [Symbol("x[$i]") for i in 1:10]
        iter_names = 101:200
        chain_names = 1:4
        x = randn(length(param_names), length(iter_names), length(chain_names))

        @testset "AxisArrays" begin
            logr = AxisArrays.AxisArray(
                x,
                AxisArrays.Axis{:param}(param_names),
                AxisArrays.Axis{:iter}(iter_names),
                AxisArrays.Axis{:chain}(chain_names),
            )
            result = psis(logr)
            @test result.log_weights isa AxisArrays.AxisArray
            @test AxisArrays.axes(result.log_weights) == AxisArrays.axes(logr)
            for k in (:pareto_shape, :tail_length, :tail_dist, :reff, :log_weights_norm)
                prop = getproperty(result, k)
                @test prop isa AxisArrays.AxisArray
                @test AxisArrays.axes(prop) == (AxisArrays.axes(logr, 1),)
            end
        end
    end
end
