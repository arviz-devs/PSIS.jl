using PSIS
using Tables
using Test

@testset "pareto_k_table" begin
    khat = [-0.1, 0.5, 0.8, 1.1, NaN, missing]
    ess = [200.0, 100.0, NaN, NaN, NaN, 50.0]

    @testset "counts" begin
        t = pareto_k_table(khat, 0.7)
        @test t.threshold == 0.7
        @test t.counts == (good=2, bad=1, very_bad=1, failed=1, missing=1)
        @test t.min_ess === nothing
        @test pareto_k_table(0.5, 0.7).counts ==
            (good=1, bad=0, very_bad=0, failed=0, missing=0)
    end

    @testset "min ESS" begin
        t = pareto_k_table(khat, 0.7; ess)
        @test keys(t.min_ess) == keys(t.counts)
        @test t.min_ess.good == 100.0
        @test all(isnan, (t.min_ess.bad, t.min_ess.very_bad, t.min_ess.failed))
        @test t.min_ess.missing == 50.0
        @test_throws DimensionMismatch pareto_k_table(khat, 0.7; ess=ess[1:2])
    end

    @testset "show" begin
        t = pareto_k_table(khat, 0.7; ess)
        @test sprint(show, MIME"text/plain"(), t) == """
            Pareto k-hat diagnostic summary:
                                      Count      Min. ESS
             (-Inf, 0.7]  (good)      2 (33.3%)  100
                (0.7, 1]  (bad)       1 (16.7%)  ——
                (1, Inf)  (very bad)  1 (16.7%)  ——
                      ——  (failed)    1 (16.7%)  ——
                      ——  (missing)   1 (16.7%)  ——"""
        @test sprint(show, MIME"text/plain"(), pareto_k_table(Float64[], 0.7)) ==
            "No parameter diagnostics."
    end

    @testset "Tables interface" begin
        t = pareto_k_table(khat, 0.7; ess)
        @test Tables.istable(t)
        cols = Tables.columntable(t)
        @test keys(cols) == (:category, :range, :count, :proportion, :min_ess)
        @test cols.category == ["good", "bad", "very bad", "failed", "missing"]
        @test isequal(cols.range, ["(-Inf, 0.7]", "(0.7, 1]", "(1, Inf)", missing, missing])
        @test cols.count == [2, 1, 1, 1, 1]
        @test cols.proportion ≈ [2, 1, 1, 1, 1] ./ 6
        @test length(Tables.rowtable(t)) == 5
        @test !haskey(Tables.columntable(pareto_k_table(khat, 0.7)), :min_ess)
    end

    @testset "results" begin
        diagnostics = ParetoDiagnostics(
            [-0.1, 0.8, 1.1, NaN], 0.9, [10.0, 100.0, Inf, NaN], [1.0, 0.4, 0.0, NaN]
        )
        @test pareto_k_table(diagnostics).threshold == 0.7
        @test pareto_k_table(diagnostics).counts ==
            (good=1, bad=1, very_bad=1, failed=1, missing=0)
        result = PSISResult(randn(100, 1, 3), [0.1, 0.6, 1.2], [80.0, NaN, NaN])
        t = pareto_k_table(result)
        @test t.threshold ≈ 0.5
        @test t.counts == (good=1, bad=1, very_bad=1, failed=0, missing=0)
        @test t.min_ess.good == 80.0
    end
end
