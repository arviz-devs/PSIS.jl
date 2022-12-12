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
function param_dim(x)
    @assert ndims(x) > 1
    return 1
end

# view of first draw from first chain (i.e. vector of parameters)
function first_draw(x::AbstractArray)
    dims = Base.setindex(ntuple(one, ndims(x)), :, param_dim(x))
    return view(x, dims...)
end

# view of all draws 
param_draws(x::AbstractArray, i::Int) = selectdim(x, param_dim(x), i)

# dimensions corresponding to draws and chains
function sample_dims(x::AbstractArray)
    d = param_dim(x)
    return filter(!=(d), ntuple(identity, ndims(x)))
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
