using PSIS
using Test

@testset "PSIS uses shared fitting and reliability diagnostics" begin
    p = ((1:100) .- 0.5) ./ 100
    logr = log.(((1 .- p) .^ (-0.8) .- 1) ./ 0.8)
    original = copy(logr)
    result = @test_logs (:warn, r"k-hat.*0.5") psis(logr)
    @test logr == original
    @test 0.5 < result.pareto_khat < 0.7
    @test isnan(result.ess_is)
    smoothed, diag = pareto_smooth(logr; is_log_scale=true)
    @test result.log_weights == smoothed
    @test result.pareto_khat == diag.khat
    @test_logs psis(logr; warn=false)
    shown = sprint(show, MIME"text/plain"(), result)
    @test occursin("bad", shown)
    @test !occursin("okay", shown)

    result = psis(collect(range(-1, 0; length=1000)); warn=false)
    @test isfinite(result.ess_is)
    @test result.ess_is ≈ PSIS.ess_is(result.log_weights, 1)
    @test maximum(result.log_weights) <= 0
    for bad in (ones(100), fill(-Inf, 100), Float64[])
        result = @test_logs psis(bad; warn=false)
        @test isequal(result.log_weights, bad)
        @test isnan(result.pareto_khat)
        @test isnan(result.ess_is)
    end
    for reff in (0, -1, NaN, Inf)
        @test_throws ArgumentError psis(logr, reff)
    end
    batched = repeat(reshape(logr, :, 1, 1), 1, 2, 3)
    batch = @test_logs psis(batched; warn=false)
    @test all(isnan, batch.ess_is)
    @test size(batch.pareto_khat) == (3,)
    @test pareto_diagnose(batch).khat == batch.pareto_khat
    @test pareto_diagnose(batch).khat_threshold == ps_khat_threshold(200)
end
