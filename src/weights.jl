"""
    $SIGNATURES

Compute importance weights from a [`PSISResult`](@ref).

# Keywords

  - `log::Bool=false`: if `true`, the log-weights are returned.
  - `normalize::Bool=true`: if `true`, the (log-)weights are normalized so that the weights
    sum to 1 along the sample dimensions.

# Returns

  - `weights`: the (log-)(normalized) importance weights with shape
    `(draws, [chains, [parameters...]])`
"""
function importance_weights(r::PSISResult; log::Bool=false, normalize::Bool=true)
    (; log_weights) = r
    # Allocate the return array ourselves to work around type instability of logsumexp
    # for DimArrays
    weights = similar(log_weights)
    if normalize
        dims = _sample_dims(log_weights)
        if log
            copyto!(weights, log_weights)
            weights .-= LogExpFunctions.logsumexp(log_weights; dims)
        else
            LogExpFunctions.softmax!(weights, log_weights; dims)
        end
    elseif log
        copyto!(weights, log_weights)
    else
        weights .= exp.(log_weights)
    end
    return weights
end

# DimArray{Float64, 3, Tuple{Dim{:draw, DimensionalData.Dimensions.Lookups.Sampled{Int64, UnitRange{Int64}, DimensionalData.Dimensions.Lookups.ForwardOrdered, DimensionalData.Dimensions.Lookups.Regular{Int64}, DimensionalData.Dimensions.Lookups.Points, DimensionalData.Dimensions.Lookups.NoMetadata}}, Dim{:chain, DimensionalData.Dimensions.Lookups.Sampled{Int64, UnitRange{Int64}, DimensionalData.Dimensions.Lookups.ForwardOrdered, DimensionalData.Dimensions.Lookups.Regular{Int64}, DimensionalData.Dimensions.Lookups.Points, DimensionalData.Dimensions.Lookups.NoMetadata}}, Dim{:param, DimensionalData.Dimensions.Lookups.Sampled{Int64, UnitRange{Int64}, DimensionalData.Dimensions.Lookups.ForwardOrdered, DimensionalData.Dimensions.Lookups.Regular{Int64}, DimensionalData.Dimensions.Lookups.Points, DimensionalData.Dimensions.Lookups.NoMetadata}}}, Tuple{}, Array{Float64, 3}, DimensionalData.NoName, DimensionalData.Dimensions.Lookups.NoMetadata}
