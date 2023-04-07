using PSIS
using Test
using AxisArrays: AxisArrays

@testset "utils" begin
    @testset "param_dim" begin
        x = randn(100, 10)
        @test PSIS.param_dim(x) == 2

        x = randn(100, 4, 10)
        @test PSIS.param_dim(x) == 3
    end

    @testset "param_draws" begin
        x = randn(100, 10)
        @test PSIS.param_draws(x, 3) === view(x, :, 3)

        x = randn(100, 4, 10)
        @test PSIS.param_draws(x, 5) === view(x, :, :, 5)
    end

    @testset "sample_dims" begin
        x = randn(100)
        @test PSIS.sample_dims(x) === Colon()
        x = randn(100, 10)
        @test PSIS.sample_dims(x) === (1,)
        x = randn(100, 4, 10)
        @test PSIS.sample_dims(x) === (1, 2)
    end

    @testset "broadcast_last_dims" begin
        adim = AxisArrays.Axis{:a}([Symbol("a[$i]") for i in 1:2])
        bdim = AxisArrays.Axis{:b}([Symbol("b[$i]") for i in 1:10])
        x = AxisArrays.AxisArray(randn(2, 10), adim, bdim)
        y = AxisArrays.AxisArray(randn(10), bdim)
        @test @inferred(PSIS.broadcast_last_dims(/, x[1, :], y)) == x[1, :] ./ y
        @test @inferred(PSIS.broadcast_last_dims(/, x, y)) == x ./ reshape(y, 1, :)
        @test @inferred(PSIS.broadcast_last_dims(/, y, x)) == reshape(y, 1, :) ./ x
        @test @inferred(PSIS.broadcast_last_dims(/, x, 3)) == x ./ 3
        @test @inferred(PSIS.broadcast_last_dims(/, 4, x)) == 4 ./ x

        @test PSIS.broadcast_last_dims(/, x, y) isa AxisArrays.AxisArray
        @test AxisArrays.axes(PSIS.broadcast_last_dims(/, x, y)) == AxisArrays.axes(x)
        @test PSIS.broadcast_last_dims(/, y, x) isa AxisArrays.AxisArray
        @test AxisArrays.axes(PSIS.broadcast_last_dims(/, y, x)) == AxisArrays.axes(x)
    end
end
