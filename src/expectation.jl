"""
    expectation(x, psis_result::PSISResult; kind=Statistics.mean)

Compute the expectation of `x` with respect to the weights in `psis_result`.

# Arguments

  - `x`: An array of values of shape `(draws[, chains[, params...]])`, to compute the
    expectation of with respect to smoothed importance weights.
  - `psis_result`: A `PSISResult` object containing the smoothed importance weights with shape
    `(draws[, chains, params...])`.

# Keywords

  - `kind=Statistics.mean`: The type of expectation to be computed. It can be any function
    that has a method for computing the weighted expectation
    `f(x::AbstractVector, weights::AbstractVector) -> Real`. In particular, the following
    are supported:

      + `Statistics.mean`
      + `Statistics.median`
      + `Statistics.std`
      + `Statistics.var`
      + `Base.Fix2(Statistics.quantile, p::Real)` for `quantile(x, weights, p)`

# Returns

  - `values`: An array of shape `(other..., params...)` or real number of `other` and `params`
    are empty containing the expectation of `x` with respect to the smoothed importance
    weights.
"""
function expectation(x::AbstractArray, psis_result::PSISResult; kind=Statistics.mean)
    log_weights = psis_result.log_weights
    weights = psis_result.weights

    param_dims = _param_dims(log_weights)
    exp_dims = _param_dims(x)
    if !isempty(exp_dims) && length(exp_dims) != length(param_dims)
        throw(
            ArgumentError(
                "The trailing dimensions of `x` must match the parameter dimensions of `psis_result.weights`",
            ),
        )
    end
    param_axes = map(Base.Fix1(axes, log_weights), param_dims)
    exp_axes = map(Base.Fix1(axes, x), exp_dims)
    if !isempty(exp_axes) && exp_axes != param_axes
        throw(
            ArgumentError(
                "The trailing axes of `x` must match the parameter axes of `psis_result.weights`",
            ),
        )
    end

    T = Base.promote_eltype(x, log_weights)
    values = similar(x, T, param_axes)

    for i in _eachparamindex(weights)
        w_i = StatsBase.AnalyticWeights(vec(_selectparam(weights, i)), 1)
        x_i = vec(ndims(x) < 3 ? x : _selectparam(x, i))
        values[i] = _expectation(kind, x_i, w_i)
    end

    iszero(ndims(values)) && return values[]

    return values
end

_expectation(f, x, weights) = f(x, weights)
function _expectation(f::Base.Fix2{typeof(Statistics.quantile),<:Real}, x, weights)
    prob = f.x
    return Statistics.quantile(x, weights, prob)
end
