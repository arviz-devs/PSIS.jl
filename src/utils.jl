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
