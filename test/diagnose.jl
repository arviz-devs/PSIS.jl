using PSIS
using Test
using StableRNGs
using Distributions: TDist
using DimensionalData: DimArray, Dimensions

@testset "sample-based Pareto diagnostics" begin
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
        @test startswith(shown, "ParetoDiagnostics with 4 parameters")
        @test occursin("Pareto k-hat diagnostic summary:", shown)
        @test occursin(r"good\s+1 \(25.0%\)", shown)
        @test occursin(r"bad\s+1 \(25.0%\)", shown)
        @test occursin(r"very bad\s+1 \(25.0%\)", shown)
        @test occursin(r"failed\s+1 \(25.0%\)", shown)
        @test !occursin("Min. ESS", shown)
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
        reff = reshape(collect(range(0.5, 1.5; length=10)), 2, 5)
        batch = @inferred pareto_diagnose(values; r_eff=reff)
        @test size(batch.khat) == (2, 5)
        @test batch.khat_threshold == pareto_khat_threshold(2000)
        @test pareto_min_ss.(batch.khat) == batch.min_ss
        @test pareto_convergence_rate.(batch.khat, 2000) == batch.convergence_rate
        for i in CartesianIndices(reff)
            onevar = pareto_diagnose(values[:, :, i]; r_eff=reff[i])
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
