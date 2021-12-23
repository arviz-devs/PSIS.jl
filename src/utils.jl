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
param_dim(x) = 1

# view of first draw from first chain (i.e. vector of parameters)
function first_draw(x::AbstractArray)
    d = ndims(x)
    @assert d > 1
    dims = Base.setindex(ntuple(one, d), :, param_dim(x))
    return view(x, dims...)
end

# view of all draws 
param_draws(x::AbstractArray, i::Int) = selectdim(x, param_dim(x), i)
