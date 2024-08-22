"""
    pareto_diagnose(x::AbstractArray; kwargs...)

Compute diagnostics for Pareto-smoothed estimate of the expectand `x`.

# Arguments

  - `x`: An array of values of shape `(draws[, chains[, params...]])`.

# Keywords

  - `reff=1`: The relative tail efficiency of `x`. Must be either a scalar or an array of
    shape `(params...,)`.
  - `is_log=false`: Whether `x` represents the log of the expectand. If `true`, the
    diagnostics are computed on the original scale, taking care to avoid numerical overflow.
  - `tails=:both`: Which tail(s) to diagnose. Valid values are `:left`, `:right`, and
    `:both`. If `tails=:both`, diagnostic values correspond to the tail with the worst
    properties.

# Returns

  - `diagnostics::ParetoDiagnostics`: A named tuple containing the following fields:

      + `pareto_shape`: The Pareto shape parameter ``k``.
      + `min_sample_size`: The minimum sample size needed for a reliable Pareto-smoothed
        estimate (i.e. to have small probability of large error).
      + `pareto_shape_threshold`: The Pareto shape ``k`` threshold needed for a reliable
        Pareto-smoothed estimate (i.e. to have small probability of large error).
      + `convergence_rate`: The relative convergence rate of the RMSE of the
        Pareto-smoothed estimate.
"""
function pareto_diagnose(
    x::AbstractArray;
    reff=1,
    is_log::Bool=false,
    tails::Union{Tails,Symbol}=BothTails,
    kind=Statistics.mean,
)
    # validate/format inputs
    _tails = _standardize_tails(tails)
    _check_requires_moments(kind)

    # diagnose the unnormalized expectation
    pareto_shape = _compute_pareto_shape(x, reff, _tails, kind, is_log)

    # compute remaining diagnostics
    sample_size = _sample_size(x)
    diagnostics = _compute_diagnostics(pareto_shape, sample_size)

    return diagnostics
end

"""
    pareto_diagnose(x::AbstractArray, ratios::AbstractArray; kwargs...)

Compute diagnostics for Pareto-smoothed importance-weighted estimate of the expectand `x`.

# Arguments

  - `x`: An array of values of shape `(draws[, chains[, params...]])`.
  - `ratios`: An array of unnormalized importance ratios of shape
    `(draws[, chains[, params...]])`.

# Keywords

  - `reff=1`: The relative efficiency of the importance weights on the original scale. Must
    be either a scalar or an array of shape `(params...,)`.
  - `is_log=false`: Whether `x` represents the log of the expectand.
  - `is_ratios_log=true`: Whether `ratios` represents the log of the importance ratios.
  - `diagnose_ratios=true`: Whether to compute diagnostics for the importance ratios.
    This should only be set to `false` if the ratios are by construction normalized, as is
    the case if they are are computed from already-normalized densities.
  - `tails`: Which tail(s) of `x * ratios` to diagnose. Valid values are `:left`, `:right`,
    and `:both`.

# Returns

  - `diagnostics::ParetoDiagnostics`: A named tuple containing the following fields:

      + `pareto_shape`: The Pareto shape parameter ``k``.
      + `min_sample_size`: The minimum sample size needed for a reliable Pareto-smoothed
        estimate (i.e. to have small probability of large error).
      + `pareto_shape_threshold`: The Pareto shape ``k`` threshold needed for a reliable
        Pareto-smoothed estimate (i.e. to have small probability of large error).
      + `convergence_rate`: The relative convergence rate of the RMSE of the
        Pareto-smoothed estimate.
"""
function pareto_diagnose(
    x::AbstractArray,
    ratios::AbstractArray{<:Real};
    is_log::Bool=false,
    is_ratios_log::Bool=true,
    diagnose_ratios::Bool=true,
    tails::Union{Tails,Symbol}=BothTails,
    kind=Statistics.mean,
    reff=1,
)

    # validate/format inputs
    _tails = _standardize_tails(tails)

    # diagnose the unnormalized expectation
    pareto_shape_numerator = if _requires_moments(kind)
        _compute_pareto_shape(x, ratios, _tails, kind, is_log, is_ratios_log)
    elseif diagnose_ratios
        nothing
    else
        throw(
            ArgumentError(
                "kind=$kind requires no moments. `diagnose_ratios` must be `true`."
            ),
        )
    end

    # diagnose the normalization term
    pareto_shape_denominator = if diagnose_ratios
        _compute_pareto_shape(ratios, reff, RightTail, Statistics.mean, is_ratios_log)
    else
        nothing
    end

    # compute the maximum of the Pareto shapes
    pareto_shape = if pareto_shape_numerator === nothing
        pareto_shape_denominator
    elseif !diagnose_ratios
        pareto_shape_numerator
    else
        max(pareto_shape_numerator, pareto_shape_denominator)
    end

    # compute remaining diagnostics
    sample_size = _sample_size(x)
    diagnostics = _compute_diagnostics(pareto_shape, sample_size)

    return diagnostics
end

# batch methods
function _compute_pareto_shape(x::AbstractArray, reff, tails::Tails, kind, is_log::Bool)
    return _map_params(x, reff) do x_i, reff_i
        return _compute_pareto_shape(x_i, reff_i, tails, kind, is_log)
    end
end
function _compute_pareto_shape(
    x::AbstractArray, r::AbstractArray, tails::Tails, kind, is_x_log::Bool, is_r_log::Bool
)
    return _map_params(x, r) do x_i, r_i
        return _compute_pareto_shape(x_i, r_i, tails, kind, is_x_log, is_r_log)
    end
end
# single methods
function _compute_pareto_shape(
    x::AbstractVecOrMat, reff::Real, tails::Tails, kind, is_log::Bool
)
    expectand_proxy = _expectand_proxy(kind, x, !is_log, is_log, is_log)
    return _compute_pareto_shape(expectand_proxy, reff, tails)
end
@constprop :aggressive function _compute_pareto_shape(
    x::AbstractVecOrMat,
    r::AbstractVecOrMat,
    tails::Tails,
    kind,
    is_x_log::Bool,
    is_r_log::Bool,
)
    expectand_proxy = _expectand_proxy(kind, x, r, is_x_log, is_r_log)
    return _compute_pareto_shape(expectand_proxy, true, tails)
end

# base method
function _compute_pareto_shape(x::AbstractVecOrMat, reff::Real, tails::Tails)
    S = length(x)
    M = _tail_length(reff, S, tails)
    T = float(eltype(x))
    if M < 5
        @warn "Tail must contain at least 5 draws. Generalized Pareto distribution cannot be reliably fit."
        return convert(T, NaN)
    end
    x_tail = similar(vec(x), M)
    return _compute_pareto_shape!(x_tail, x, tails)
end

function _compute_pareto_shape!(x_tail::AbstractVector, x::AbstractVecOrMat, tails::Tails)
    _tails = tails === BothTails ? (LeftTail, RightTail) : (tails,)
    return maximum(_tails) do tail
        tail_dist = _fit_tail_dist!(x_tail, x, tail)
        return pareto_shape(tail_dist)
    end
end

function _fit_tail_dist!(x_tail, x, tail)
    M = length(x_tail)
    x_tail_view, cutoff = _tail_and_cutoff(vec(x), M, tail)
    if any(!isfinite, x_tail_view)
        @warn "Tail contains non-finite values. Generalized Pareto distribution cannot be reliably fit."
        T = float(eltype(x_tail))
        return GeneralizedPareto(zero(T), convert(T, NaN), convert(T, NaN))
    end
    _shift_tail!(x_tail, x_tail_view, cutoff, tail)
    return fit_gpd(x_tail; prior_adjusted=true, sorted=true)
end
