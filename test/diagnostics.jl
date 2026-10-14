using PSIS
using Test
using StableRNGs
using Distributions: TDist
using DimensionalData: DimArray, Dimensions

@testset "Pareto diagnostics" begin
    @testset "scalar diagnostics" begin
        @testset "pareto_min_ss" begin
            @test PSIS.pareto_min_ss(-1.0) == 10
            @test PSIS.pareto_min_ss(0.0) == 10
            @test PSIS.pareto_min_ss(0.5) ≈ 100
            @test PSIS.pareto_min_ss(1.0) == Inf
            @test PSIS.pareto_min_ss(2.0) == Inf
            @test isnan(PSIS.pareto_min_ss(NaN))
            @test PSIS.pareto_min_ss(0.5f0) isa Float32
            @test PSIS.pareto_min_ss(1//2) isa AbstractFloat
        end

        @testset "pareto_khat_threshold" begin
            @test PSIS.pareto_khat_threshold(100) == 0.5
            @test PSIS.pareto_khat_threshold(10_000) == 0.75
            for n in (0, 1, 9, 9.5, 10)
                @test PSIS.pareto_khat_threshold(n) == 0
            end
            @test PSIS.pareto_khat_threshold(9.0f0) === 0.0f0
            @test isnan(PSIS.pareto_khat_threshold(NaN))
            for k in (0.0, 0.3, 0.5, 0.7)
                @test PSIS.pareto_khat_threshold(PSIS.pareto_min_ss(k)) ≈ k
            end
        end

        @testset "pareto_convergence_rate" begin
            @test PSIS.pareto_convergence_rate(-1.0, 1000) == 1
            @test PSIS.pareto_convergence_rate(0.0, 1000) == 1
            @test PSIS.pareto_convergence_rate(1.0, 1000) == 0
            @test PSIS.pareto_convergence_rate(2.0, 1000) == 0
            @test isnan(PSIS.pareto_convergence_rate(NaN, 1000))
            for n in (-1, 0, 1), k in (-1.0, 0.5, 1.0)
                @test isnan(PSIS.pareto_convergence_rate(k, n))
            end
            # The continuous limit of the smooth expression includes 1/(n-1).
            @test PSIS.pareto_convergence_rate(0.5, 1000) ≈ 0.8562361736999171
            for k in (0.01, 0.3, 0.5 - 1e-10, 0.5 + 1e-10, 0.7, 0.99)
                # Independently evaluate the original Appendix B expression at high precision.
                expected = setprecision(256) do
                    b, n = BigFloat(k), big"1000"
                    (2(b - 1)*n^(2b + 1) + (1 - 2b)*n^(2b) + n^2) / ((n - 1)*(n - n^(2b)))
                end
                @test PSIS.pareto_convergence_rate(k, 1000) ≈ Float64(expected) atol=1e-14
            end
            for k in (0.1, 0.5, 0.9)
                @test isfinite(PSIS.pareto_convergence_rate(k, floatmax(Float64)))
                @test 0 ≤ PSIS.pareto_convergence_rate(k, 1000) ≤ 1
            end
            @test PSIS.pareto_convergence_rate(0.5f0, 1000) isa Float32
            @test PSIS.pareto_convergence_rate(1//2, 1000) isa AbstractFloat
        end
    end

    @testset "diagnostics computed from samples" begin
        for n in (0, 1, 9, 10)
            values = zeros(n)
            @test pareto_khat_threshold(values) == 0
            @test pareto_diagnose(values).khat_threshold == 0
        end
        rng = StableRNG(18)
        x = rand(rng, TDist(3), 10_000, 2)
        original = copy(x)
        d = @inferred pareto_diagnose(x)
        @test x == original
        @test d isa ParetoDiagnostics
        @test d.khat ≈ 1/3 atol=0.1
        @test (@inferred pareto_khat(x)) == d.khat
        @test pareto_min_ss(x) == d.min_ss
        @test pareto_khat_threshold(x) == d.khat_threshold
        @test pareto_convergence_rate(x) == d.convergence_rate
        @test pareto_min_ss(d.khat) == d.min_ss
        @test pareto_khat_threshold(length(x)) == d.khat_threshold
        @test pareto_convergence_rate(d.khat, length(x)) == d.convergence_rate
        @test pareto_khat(vec(x)) == d.khat
        @test pareto_khat(x) == max(pareto_khat(x; tail=:left), pareto_khat(x; tail=:right))
        @test pareto_khat(-x; tail=:left) ≈ pareto_khat(x; tail=:right)
        positive = abs.(x)
        @test pareto_khat(log.(positive); is_log_scale=true) ≈
            pareto_khat(positive; tail=:right)
        @test_throws ArgumentError pareto_khat(x; tail=:invalid)
        @test_throws ArgumentError pareto_khat(x; is_log_scale=true, tail=:both)
        @test_throws ArgumentError pareto_khat(x; r_eff=0)
        @test_throws DimensionMismatch pareto_khat(x; r_eff=[1, 2])
        @test pareto_khat(x; r_eff=[0.8]) == pareto_khat(x; r_eff=0.8)
        @test pareto_khat(x; tail_length=100) isa Float64
        @test_throws ArgumentError pareto_khat(x; tail_length=4)
        @test_throws ArgumentError pareto_khat(x; tail_length=length(x))
        @test_throws ArgumentError pareto_khat(x; tail_length=length(x) ÷ 2)

        @testset "show" begin
            diagnostics = ParetoDiagnostics(
                [-0.1, 0.8, 1.1, NaN], 0.7, [10.0, 100.0, Inf, NaN], [1.0, 0.4, 0.0, NaN]
            )
            shown = sprint(show, MIME"text/plain"(), diagnostics)
            @test shown == """
                ParetoDiagnostics with 4 parameters
                Pareto k-hat diagnostic summary:
                                          Count
                 (-Inf, 0.7]  (good)      1 (25.0%)
                    (0.7, 1]  (bad)       1 (25.0%)
                    (1, Inf)  (very bad)  1 (25.0%)
                          ——  (failed)    1 (25.0%)"""
        end

        @testset "failure is quiet and explicit" begin
            for values in
                (Float64[], [1.0], collect(1.0:20), ones(100), [NaN; x[:]], [Inf; x[:]])
                @test_logs d = pareto_diagnose(values)
                @test isnan(d.khat)
                @test isnan(d.min_ss)
                @test isnan(d.convergence_rate)
            end
            @test_logs (:warn, r"could not be fit") pareto_khat(ones(100); warn=true)
            @test isnan(pareto_khat([ones(500); collect(2.0:501)]; tail=:both))
        end

        @testset "parameter batches" begin
            values = reshape(x, 1000, 2, 2, 5)
            r_eff = reshape(collect(range(0.5, 1.5; length=10)), 2, 5)
            batch = @inferred pareto_diagnose(values; r_eff)
            @test size(batch.khat) == (2, 5)
            @test batch.khat_threshold == pareto_khat_threshold(2000)
            @test pareto_min_ss.(batch.khat) == batch.min_ss
            @test pareto_convergence_rate.(batch.khat, 2000) == batch.convergence_rate
            for i in CartesianIndices(r_eff)
                onevar = pareto_diagnose(values[:, :, i]; r_eff=r_eff[i])
                @test batch.khat[i] == onevar.khat
                @test batch.min_ss[i] == onevar.min_ss
                @test batch.convergence_rate[i] == onevar.convergence_rate
            end
            @test_throws DimensionMismatch pareto_khat(values; r_eff=ones(10))
            @test_throws ArgumentError pareto_khat(values; r_eff=fill(NaN, 2, 5))
            @test pareto_khat(Float32.(values)) isa Array{Float32,2}
            named = DimArray(
                reshape(x, 1000, 2, 10),
                (
                    Dimensions.Dim{:draw}(1:1000),
                    Dimensions.Dim{:chain}(1:2),
                    Dimensions.Dim{:param}(Symbol.("x", 1:10)),
                ),
            )
            diag = @inferred pareto_diagnose(named)
            @test diag.khat isa DimArray
            @test Dimensions.dims(diag.khat) == Dimensions.dims(named, (:param,))
        end
    end
end
