using PSIS
using Test
using DimensionalData: Dimensions, DimArray

@testset "utils" begin
    @testset "_param_dims" begin
        x = randn(100)
        @test PSIS._param_dims(x) == ()

        x = randn(100, 10)
        @test PSIS._param_dims(x) == ()

        x = randn(100, 4, 10)
        @test PSIS._param_dims(x) == (3,)

        x = randn(100, 4, 5, 10)
        @test PSIS._param_dims(x) == (3, 4)

        x = randn(100, 4, 5, 6, 10)
        @test PSIS._param_dims(x) == (3, 4, 5)
    end

    @testset "_eachparamindex/_selectparam" begin
        x = randn(100)
        @test size(PSIS._eachparamindex(x)) == ()
        @test PSIS._selectparam(x, PSIS._eachparamindex(x)[1]) == x

        x = randn(100, 4)
        @test size(PSIS._eachparamindex(x)) == ()
        @test PSIS._selectparam(x, PSIS._eachparamindex(x)[1]) == x

        x = randn(100, 4, 5)
        @test size(PSIS._eachparamindex(x)) == (5,)
        @test PSIS._selectparam.(Ref(x), PSIS._eachparamindex(x)) ==
            collect(eachslice(x; dims=3))

        x = randn(100, 4, 5, 3)
        @test size(PSIS._eachparamindex(x)) == (5, 3)
        if VERSION ≥ v"1.9"
            @test PSIS._selectparam.(Ref(x), PSIS._eachparamindex(x)) ==
                eachslice(x; dims=(3, 4))
        end
    end

    @testset "_sample_dims" begin
        x = randn(100)
        @test PSIS._sample_dims(x) === (1,)
        x = randn(100, 10)
        @test PSIS._sample_dims(x) === (1, 2)
        x = randn(100, 4, 10)
        @test PSIS._sample_dims(x) === (1, 2)
    end
end
