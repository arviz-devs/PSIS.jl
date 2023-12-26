@enum Tails LeftTail RightTail BothTails
const TAIL_OPTIONS = (left=LeftTail, right=RightTail, both=BothTails)

_validate_tails(tails::Tails) = tails
function _validate_tails(tails::Symbol)
    if !haskey(TAIL_OPTIONS, tails)
        throw(ArgumentError("invalid tails: $tails. Valid values are :left, :right, :both"))
    end
    return TAIL_OPTIONS[tails]
end

_default_tails(log::Bool) = log ? RightTail : BothTails

_as_scale(log::Bool) = log ? Base.log : identity

"""
    pareto_diagnose(x::AbstractArray; warn=false, reff=1, log=false[, tails::Symbol])

Compute diagnostics for Pareto-smoothed estimate of the expectand ``x``.

# Arguments

  - `x`: An array of values of shape `(draws[, chains[, params...]])`.

# Keywords

  - `warn=false`: Whether to raise an informative warning if the diagnostics indicate that the
    Pareto-smoothed estimate may be unreliable.
  - `reff=1`: The relative efficiency of the importance weights. Must be either a scalar or an
    array of shape `(params...,)`.
  - `log=false`: Whether `x` represents the log of the expectand. If `true`, the diagnostics
    are computed on the original scale, taking care to avoid numerical overflow.
  - `tails`: Which tail(s) to use for the diagnostics. Valid values are `:left`, `:right` and
    `:both`. If `log=true`, only `:right` is valid. Defaults to `:both` if `log=false`.

# Returns

  - `diagnostics::NamedTuple`: A named tuple containing the following fields:

      + `pareto_shape`: The Pareto shape parameter ``k``.
      + `min_sample_size`: The minimum sample size needed for a reliable Pareto-smoothed
        estimate (i.e. to have small probability of large error).
      + `pareto_shape_threshold`: The Pareto shape ``k`` threshold needed for a reliable
        Pareto-smoothed estimate (i.e. to have small probability of large error).
      + `convergence_rate`: The relative convergence rate of the RMSE of the
        Pareto-smoothed estimate.
"""
function pareto_diagnose(
    x::AbstractArray{<:Real};
    warn::Bool=false,
    reff=1,
    log::Bool=false,
    tails::Union{Tails,Symbol}=_default_tails(log),
)
    _tails = _validate_tails(tails)
    if log && _tails !== RightTail
        throw(ArgumentError("log can only be true when tails=:right"))
    end
    sample_size = prod(map(Base.Fix1(size, x), _sample_dims(x)))
    if _tails === BothTails
    end
    diagnostics = _pareto_diagnose(x, reff, _tails, _as_scale(log))
    if warn
        # TODO: check diagnostics and raise warning
    end
    return diagnostics
end

function _pareto_diagnose(x::AbstractArray, reff, tails::Tails, scale)
    tail_dist = _fit_tail_dist(x, reff, tails, scale)
    sample_size = prod(map(Base.Fix1(size, x), _sample_dims(x)))
    return _compute_diagnostics(pareto_shape(tail_dist), sample_size)
end

function _compute_diagnostics(pareto_shape, sample_size)
    return (
        pareto_shape,
        min_sample_size=min_sample_size(pareto_shape),
        pareto_shape_threshold=pareto_shape_threshold(sample_size),
        convergence_rate=convergence_rate(pareto_shape, sample_size),
    )
end

@inline function _fit_tail_dist(
    x::AbstractArray,
    reff::Union{Real,AbstractArray{<:Real}},
    tails::Tails,
    scale::Union{typeof(log),typeof(identity)},
)
    return map(_eachparamindex(x)) do i
        reff_i = reff isa Real ? reff : _selectparam(reff, i)
        return _fit_tail_dist(_selectparam(x, i), reff_i, tails, scale)
    end
end
function _fit_tail_dist(
    x::AbstractVecOrMat,
    reff::Real,
    tails::Tails,
    scale::Union{typeof(log),typeof(identity)},
)
    S = length(x)
    M = tail_length(reff, S)
    x_tail = similar(vec(x), M)
    _tails = tails === BothTails ? (LeftTail, RightTail) : (tails,)
    tail_dists = map(_tails) do tail
        _, cutoff = _get_tail!(x_tail, vec(x), tail)
        _shift_tail!(x_tail, cutoff, tail, scale)
        return _fit_tail_dist(x_tail)
    end
    tail_dist = argmax(pareto_shape, tail_dists)
    return tail_dist
end
_fit_tail_dist(x_tail::AbstractVector) = fit_gpd(x_tail; prior_adjusted=true, sorted=true)

function _get_tail!(x_tail::AbstractVector, x::AbstractVector, tail::Tails)
    S = length(x)
    M = length(x_tail)
    ind_offset = firstindex(x) - 1
    perm = partialsortperm(x, ind_offset .+ ((S - M):S); rev=tail === LeftTail)
    cutoff = x[first(perm)]
    tail_inds = @view perm[(firstindex(perm) + 1):end]
    copyto!(x_tail, @views x[tail_inds])
    return x_tail, cutoff
end

function _shift_tail!(
    x_tail, cutoff, tails::Tails, scale::Union{typeof(log),typeof(identity)}
)
    if scale === log
        x_tail_max = x_tail[end]
        @. x_tail = exp(x_tail - x_tail_max) - exp(cutoff - x_tail_max)
    elseif tails === LeftTail
        @. x_tail = cutoff - x_tail
    else
        @. x_tail = x_tail - cutoff
    end
    return x_tail
end
