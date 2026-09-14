using PSIS
using Test

@testset "diagnostics from k and sample size" begin
    @test PSIS.ps_min_ss(-1.0) == 10
    @test PSIS.ps_min_ss(0.0) == 10
    @test PSIS.ps_min_ss(0.5) ≈ 100
    @test PSIS.ps_min_ss(1.0) == Inf
    @test PSIS.ps_min_ss(2.0) == Inf
    @test isnan(PSIS.ps_min_ss(NaN))
    @test PSIS.ps_khat_threshold(100) == 0.5
    @test PSIS.ps_khat_threshold(10_000) == 0.75
    @test_throws ArgumentError PSIS.ps_khat_threshold(1)
    @test_throws ArgumentError PSIS.ps_khat_threshold(0)

    @test PSIS.ps_convergence_rate(-1.0, 1000) == 1
    @test PSIS.ps_convergence_rate(0.0, 1000) == 1
    @test PSIS.ps_convergence_rate(1.0, 1000) == 0
    @test PSIS.ps_convergence_rate(2.0, 1000) == 0
    @test isnan(PSIS.ps_convergence_rate(NaN, 1000))
    @test_throws ArgumentError PSIS.ps_convergence_rate(0.5, 1)
    # The continuous limit of the smooth expression includes 1/(n-1).
    @test PSIS.ps_convergence_rate(0.5, 1000) ≈ 0.8562361736999171
    for k in (0.01, 0.3, 0.5 - 1e-10, 0.5 + 1e-10, 0.7, 0.99)
        # Independently evaluate the original Appendix B expression at high precision.
        expected = setprecision(256) do
            b, n = BigFloat(k), big"1000"
            (2(b - 1)*n^(2b + 1) + (1 - 2b)*n^(2b) + n^2) / ((n - 1)*(n - n^(2b)))
        end
        @test PSIS.ps_convergence_rate(k, 1000) ≈ Float64(expected) atol=1e-14
    end
    for k in (0.1, 0.5, 0.9)
        @test isfinite(PSIS.ps_convergence_rate(k, floatmax(Float64)))
        @test 0 ≤ PSIS.ps_convergence_rate(k, 1000) ≤ 1
    end
    @test PSIS.ps_min_ss(0.5f0) isa Float32
    @test PSIS.ps_convergence_rate(0.5f0, 1000) isa Float32
end
