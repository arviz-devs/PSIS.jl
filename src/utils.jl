function uniform_probabilities(T, npoints::Integer)
    pmin = 1 / (2 * T(npoints))
    return pmin:(1 / T(npoints)):(1 - pmin)
end

as_array(x::AbstractArray) = x
as_array(x) = [x]

missing_to_nan(x::AbstractArray{>:Missing}) = replace(x, missing => NaN)
missing_to_nan(::Missing) = NaN
missing_to_nan(x) = x

# dimensions corresponding to draws (and maybe chains)
_sample_dims(x::AbstractArray) = ntuple(identity, min(2, ndims(x)))

# dimension corresponding to parameters
_param_dims(x::AbstractArray) = ntuple(i -> i + 2, max(0, ndims(x) - 2))

# axes corresponding to parameters
_param_axes(x::AbstractArray) = map(Base.Fix1(axes, x), _param_dims(x))

# iterate over all parameters; combine with _selectparam
_eachparamindex(x::AbstractArray) = CartesianIndices(_param_axes(x))

# view of all draws for a param
function _selectparam(x::AbstractArray, i::CartesianIndex)
    sample_dims = ntuple(_ -> Colon(), ndims(x) - length(i))
    return view(x, sample_dims..., i)
end

function _maybe_log_normalize!(x::AbstractArray, normalize::Bool)
    if normalize
        x .-= LogExpFunctions.logsumexp(x; dims=_sample_dims(x))
    end
    return x
end
