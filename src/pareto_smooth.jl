"""
    pareto_smooth(x::AbstractArray; kwargs...)

Pareto-smooth the values `x` for computation of the mean.

# Arguments

  - `x`: An array of values of shape `(draws[, chains[, params...]])`.

# Keywords

  - `reff=1`: The relative tail efficiency of `x`. Must be either a scalar or an array of
    shape `(params...,)`.
  - `is_log=false`: Whether `x` represents the log of the expectand. If `true`, the
    diagnostics are computed on the original scale, taking care to avoid numerical overflow.
  - `tails=:both`: Which tail(s) to smooth. Valid values are `:left`, `:right`, and
    `:both`. If `tails=:both`, diagnostic values correspond to the tail with the worst
    properties.

# Returns

  - `x_smoothed`: An array of the same shape as `x` with the specified tails Pareto-
    smoothed.
  - `diagnostics::ParetoDiagnostics`: Pareto diagnostics for the specified tails.
"""
function pareto_smooth(
    x::AbstractArray{<:Real};
    reff=1,
    is_log::Bool=false,
    tails::Union{Tails,Symbol}=BothTails,
    warn::Bool=true,
)
    # validate/format inputs
    _tails = _standardize_tails(tails)

    # smooth the tails and compute Pareto shape
    x_smooth, pareto_shape = _pareto_smooth(x, reff, _tails, is_log)

    # compute remaining diagnostics
    diagnostics = _compute_diagnostics(pareto_shape, _sample_size(x))

    # warn if necessary
    warn && check_pareto_diagnostics(diagnostics)

    return x_smooth, diagnostics
end

function _pareto_smooth(x, reff, tails, is_log)
    x_smooth = similar(x, float(eltype(x)))
    copyto!(x_smooth, x)
    pareto_shape = _pareto_smooth!(x_smooth, reff, tails, is_log)
    return x_smooth, pareto_shape
end

function _pareto_smooth!(x::AbstractArray, reff, tails::Tails, is_log::Bool)
    # workaround for mysterious type-non-inferrability for 3d arrays
    T = typeof(float(one(eltype(x))))
    pareto_shape = similar(x, T, _param_axes(x))
    copyto!(
        pareto_shape,
        _map_params(x, reff) do x_i, reff_i
            _pareto_smooth!(x_i, reff_i, tails, is_log)
        end,
    )
    return pareto_shape
end
function _pareto_smooth!(x::AbstractVecOrMat, reff::Real, tails::Tails, is_log::Bool)
    M = _tail_length(reff, length(x), tails)
    if tails == BothTails
        return max(
            _pareto_smooth_tail_of_length!(x, M, LeftTail, is_log),
            _pareto_smooth_tail_of_length!(x, M, RightTail, is_log),
        )
    else
        return _pareto_smooth_tail_of_length!(x, M, tails, is_log)
    end
end

# this function barrier is necessary to avoid type instability
function _pareto_smooth_tail_of_length!(x, tail_length, tail, is_log)
    x_tail, cutoff = _tail_and_cutoff(vec(x), tail_length, tail)
    dist = _fit_tail_dist_and_smooth!(x_tail, cutoff, tail, is_log)
    return pareto_shape(dist)
end

function _fit_tail_dist_and_smooth!(x_tail, cutoff, tail, is_log)
    if is_log
        x_max = tail === LeftTail ? cutoff : last(x_tail)
        x_tail .= exp.(x_tail .- x_max)
        cutoff = exp(cutoff - x_max)
    end
    _shift_tail!(x_tail, x_tail, cutoff, tail)
    dist = fit_gpd(x_tail; prior_adjusted=true, sorted=true)
    _pareto_smooth_tail!(x_tail, dist)
    _shift_tail!(x_tail, x_tail, tail === RightTail ? -cutoff : cutoff, tail)
    if is_log
        x_tail .= min.(log.(x_tail), 0) .+ x_max
    end
    return dist
end

function _pareto_smooth_tail!(x_tail, tail_dist)
    p = uniform_probabilities(eltype(x_tail), length(x_tail))
    x_tail .= quantile.(Ref(tail_dist), p)
    return x_tail
end
