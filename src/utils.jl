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

"""
    broadcast_last_dims(f, x, y)

Compute `f.(x, y)` but where `y` shares the last dimensions of `x` instead of the first.

This function adds leading singleton dimensions to `y` until it has the same number of
dimensions as `x`.

The function tries to keep the final array type as close as possible to the input type.
"""
function broadcast_last_dims(f, x, y)
    (x isa Number || y isa Number || ndims(x) == ndims(y)) && return f.(x, y)
    if ndims(x) > ndims(y)
        yreshape = reshape(y, ntuple(one, ndims(x) - ndims(y))..., size(y)...)
        z = f.(x, yreshape)
        zdim = similar(x, eltype(z))
    else
        xreshape = reshape(x, ntuple(one, ndims(y) - ndims(x))..., size(x)...)
        z = f.(xreshape, y)
        zdim = similar(y, eltype(z))
    end
    copyto!(zdim, z)
    return zdim
end
