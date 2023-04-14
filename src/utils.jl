function uniform_probabilities(T, npoints::Integer)
    pmin = 1 / (2 * T(npoints))
    return pmin:(1 / T(npoints)):(1 - pmin)
end

as_array(x::AbstractArray) = x
as_array(x) = [x]

missing_to_nan(x::AbstractArray{>:Missing}) = replace(x, missing => NaN)
missing_to_nan(::Missing) = NaN
missing_to_nan(x) = x

# dimension corresponding to parameters
function param_dims(x)
    N = ndims(x)
    @assert N > 1
    N == 2 && return (2,)
    N ≥ 3 && return ntuple(i -> i + 2, N - 2)
end

# view of all draws
function param_draws(x::AbstractArray, i::CartesianIndex)
    sample_dims = ntuple(_ -> Colon(), ndims(x) - length(i))
    return view(x, sample_dims..., i)
end

# dimensions corresponding to draws and chains
function sample_dims(x::AbstractArray)
    d = param_dims(x)
    return filter(∉(d), ntuple(identity, ndims(x)))
end
sample_dims(::AbstractVector) = Colon()

function _maybe_log_normalize!(x::AbstractArray, normalize::Bool)
    if normalize
        x .-= LogExpFunctions.logsumexp(x; dims=sample_dims(x))
    end
    return x
end
