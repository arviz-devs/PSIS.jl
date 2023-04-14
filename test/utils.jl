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
end
