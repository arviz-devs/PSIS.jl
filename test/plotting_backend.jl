using PSIS
using Test

@testset "plotting backend" begin
    PSIS._paretoshapeplot(::Val{:TestBackend}, args...; kwargs...) = 10
    PSIS._paretoshapeplot!(::Val{:TestBackend}, args...; kwargs...) = 11
    old_backend = PSIS.PLOTTING_BACKEND[]
    PSIS.plotting_backend!(:TestBackend)
    @test PSIS.PLOTTING_BACKEND[] == :TestBackend
    values = randn(100)
    @test paretoshapeplot(values) === 10
    @test paretoshapeplot!(values) === 11
    PSIS.plotting_backend!(old_backend)
end
