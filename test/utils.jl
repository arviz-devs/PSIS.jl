using PSIS
using Test

@testset "utils" begin
    @testset "param_dim" begin
        x = randn(10, 100)
        @test PSIS.param_dim(x) == 1

        x = randn(10, 100, 4)
        @test PSIS.param_dim(x) == 1
    end

    @testset "first_draw" begin
        x = randn(10, 100)
        @test PSIS.first_draw(x) === view(x, :, 1)

        x = randn(10, 100, 4)
        @test PSIS.first_draw(x) === view(x, :, 1, 1)
    end

    @testset "param_draws" begin
        x = randn(10, 100)
        @test PSIS.param_draws(x, 3) === view(x, 3, :)

        x = randn(10, 100, 4)
        @test PSIS.param_draws(x, 5) === view(x, 5, :, :)
    end

    @testset "sample_dims" begin
        x = randn(100)
        @test PSIS.sample_dims(x) === Colon()
        x = randn(10, 100)
        @test PSIS.sample_dims(x) === (2,)
        x = randn(10, 100, 4)
        @test PSIS.sample_dims(x) === (2, 3)
    end
end
