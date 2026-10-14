using PSIS
using Test

@testset "tail fitting without smoothing" begin
    # GPD(k=0.5, sigma=1) quantiles provide a deterministic heavy right tail.
    x = [(1 - (i - 0.5) / 1000)^(-0.5) * 2 - 2 for i in 1:1000]
    original = copy(x)
    fit = PSIS._fit_tail(x, 94; tail=:right, is_log_scale=false)
    @test x == original
    @test fit.status === :ok
    @test fit.dist.k ≈ 0.5 atol=0.1
    @test length(fit.indices) == 94
    reflected = PSIS._fit_tail(-x, 94; tail=:left, is_log_scale=false)
    @test reflected.dist.k ≈ fit.dist.k
    logfit = PSIS._fit_tail(log.(x), 94; tail=:right, is_log_scale=true)
    @test logfit.dist.k ≈ fit.dist.k
    shifted = PSIS._fit_tail(log.(x) .+ 1500, 94; tail=:right, is_log_scale=true)
    @test shifted.dist.k ≈ fit.dist.k atol=1e-10
    @test PSIS._fit_tail(x, 4; tail=:right, is_log_scale=false).status === :insufficient
    @test PSIS._fit_tail(fill(1.0, 100), 20; tail=:right, is_log_scale=false).status ===
        :constant
    for bad in (NaN, Inf, -Inf)
        invalid = [bad; x]
        @test PSIS._fit_tail(invalid, 94; tail=:right, is_log_scale=false).status ===
            :nonfinite
    end
    zeros_and_tail = [fill(-Inf, 900); log.(x[901:end])]
    @test PSIS._fit_tail(zeros_and_tail, 94; tail=:right, is_log_scale=true).status === :ok
end

@testset "v9 tail length" begin
    @test PSIS._tail_length(1000, 1) == 94
    @test PSIS._tail_length(1000, 0.001) == 200
    @test PSIS._tail_length(1000, 100) == 9
    @test PSIS._tail_length(20, 1) == 4
    @test PSIS._tail_length(0, 1) == 0
    for bad in (0, -1, NaN, Inf)
        @test_throws ArgumentError PSIS._tail_length(1000, bad)
    end
end
