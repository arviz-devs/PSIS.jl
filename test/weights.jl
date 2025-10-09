using DimensionalData
using LogExpFunctions
using PSIS
using Test

@testset "importance_weights" begin
    @testset "Array" begin
        sz = (100, 4, 3, 2)
        @testset for n_axes in eachindex(sz)
            _sz = sz[1:n_axes]
            dims = Tuple(first(1:n_axes, 2))
            log_weights = randn(_sz)
            reff = rand(_sz[3:end]...)
            pareto_shape = rand(_sz[3:end]...)
            result = PSISResult(log_weights, pareto_shape, reff)
            log_weights_norm = logsumexp(log_weights; dims)
            log_weights_normalized = log_weights .- log_weights_norm
            weights_normalized = exp.(log_weights_normalized)
            @test @inferred(importance_weights(result)) ==
                importance_weights(result; normalize=true, log=false)
            @test importance_weights(result; log=false, normalize=true) ≈ weights_normalized
            @test importance_weights(result; log=true, normalize=true) ≈
                log_weights_normalized
            @test importance_weights(result; log=false, normalize=false) ≈ exp.(log_weights)
            @test importance_weights(result; log=true, normalize=false) == log_weights
        end
    end

    @testset "compatibility with custom axes" begin
        @testset "DimensionalData" begin
            sample_axes = (Dimensions.Dim{:draw}(1:100), Dimensions.Dim{:chain}(1:4))
            param_axes = (Dimensions.Dim{:param1}(1:3), Dimensions.Dim{:param2}(1:2))
            @testset for n_param_dims in 0:2
                _param_axes = param_axes[1:n_param_dims]
                log_weights = rand(sample_axes..., _param_axes...)
                @assert log_weights isa DimArray
                reff = rand(_param_axes...)
                pareto_shape = rand(_param_axes...)
                result = PSISResult(log_weights, pareto_shape, reff)
                result_array = PSISResult(collect.((log_weights, pareto_shape, reff))...)
                @test @inferred(importance_weights(result)) isa DimArray
                @testset for normalize in (true, false), log in (true, false)
                    weights = importance_weights(result; normalize, log)
                    @test weights ≈ importance_weights(result_array; normalize, log)
                    @test weights isa DimArray
                    @test Dimensions.dims(weights) == Dimensions.dims(log_weights)
                end
            end
        end
    end
end
