using PSIS
using Test
using Random
using ReferenceTests
using Distributions: Normal, Cauchy, Exponential, logpdf, mean
using LogExpFunctions: softmax
using Logging: SimpleLogger, with_logger
using AxisArrays: AxisArrays

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
                r_eff = length(sz) == 1 ? 1.0 : ones(sz[1])
                rng = MersenneTwister(42)
                x = rand(rng, proposal, sz)
                logr = logpdf.(target, x) .- logpdf.(proposal, x)

                logw, k = psis(logr, r_eff)
                w = softmax(logr; dims=dims)
                @test all(≈(ξ_exp; atol=0.15), k)
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
            @testset for sz in (100, (5, 100), (5, 100, 4))
                dims = length(sz) == 1 ? Colon() : 2:length(sz)
                r_eff = length(sz) == 1 ? 1.0 : ones(sz[1])
                x = randn(sz)
                lw1, k1 = psis(x, r_eff)
                lw2, k2 = psis(x, r_eff; normalize=true)
                @test k1 ≈ k2
                @test !(lw1 ≈ lw2)

                @test all(abs.(diff(lw1 .- lw2; dims=length(sz))) .< sqrt(eps()))
                @test all(≈(1), sum(exp, lw2; dims=dims))
            end
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
        sz = (5, 1_000, 4)
        x = rand(rng, proposal, sz)
        logr = logpdf.(target, x) .- logpdf.(proposal, x)
        expected_khats = Dict(
            (0.7, false) => [0.45848943, 0.73939023, 0.64318395, 0.8255847, 0.87575057],
            (1.2, false) => [0.42288872, 0.6686345, 0.73749322, 0.76318927, 0.83505587],
            (0.7, true) => [0.45334008, 0.74012806, 0.64558096, 0.82759211, 0.8813605],
            (1.2, true) => [0.4225601, 0.67035541, 0.74046699, 0.76625258, 0.8395082],
        )
        @testset for r_eff in (0.7, 1.2), improved in (true, false)
            r_effs = fill(r_eff, sz[1])
            logw, k = psis(logr, r_effs; improved=improved)
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
            r_eff = ones(10)
            logw, k = psis(logr, r_eff)
            @test logw isa AxisArrays.AxisArray
            @test AxisArrays.axes(logw) == AxisArrays.axes(logr)
            @test k isa AxisArrays.AxisArray
            @test AxisArrays.axes(k) == (AxisArrays.axes(logr, 1),)
        end
    end
end
