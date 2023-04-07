using PSIS
using Test
using DimensionalData: Dimensions, DimArray

@testset "utils" begin
    @testset "param_dim" begin
        x = randn(100, 10)
        @test PSIS.param_dims(x) == (2,)

        x = randn(100, 4, 10)
        @test PSIS.param_dims(x) == (3,)

        x = randn(100, 4, 5, 10)
        @test PSIS.param_dims(x) == (3, 4)

        x = randn(100, 4, 5, 6, 10)
        @test PSIS.param_dims(x) == (3, 4, 5)
    end

    @testset "param_draws" begin
        x = randn(100, 10)
        @test PSIS.param_draws(x, CartesianIndex(3)) === view(x, :, 3)

        x = randn(100, 4, 10)
        @test PSIS.param_draws(x, CartesianIndex(5)) === view(x, :, :, 5)

        x = randn(100, 4, 5, 10)
        @test PSIS.param_draws(x, CartesianIndex(5, 6)) === view(x, :, :, 5, 6)

        x = randn(100, 4, 5, 6, 10)
        @test PSIS.param_draws(x, CartesianIndex(5, 6, 7)) === view(x, :, :, 5, 6, 7)
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
        adim = Dimensions.Dim{:a}([Symbol("a[$i]") for i in 1:2])
        bdim = Dimensions.Dim{:b}([Symbol("b[$i]") for i in 1:10])
        x = DimArray(randn(2, 10), (adim, bdim))
        y = DimArray(randn(10), (bdim,))
        @test @inferred(PSIS.broadcast_last_dims(/, x[1, :], y)) == x[1, :] ./ y
        @test @inferred(PSIS.broadcast_last_dims(/, x, y)) == x ./ reshape(y, 1, :)
        @test @inferred(PSIS.broadcast_last_dims(/, y, x)) == reshape(y, 1, :) ./ x
        @test @inferred(PSIS.broadcast_last_dims(/, x, 3)) == x ./ 3
        @test @inferred(PSIS.broadcast_last_dims(/, 4, x)) == 4 ./ x

        @test PSIS.broadcast_last_dims(/, x, y) isa DimArray
        @test Dimensions.dims(PSIS.broadcast_last_dims(/, x, y)) == Dimensions.dims(x)
        @test PSIS.broadcast_last_dims(/, y, x) isa DimArray
        @test Dimensions.dims(PSIS.broadcast_last_dims(/, y, x)) == Dimensions.dims(x)
    end
end
