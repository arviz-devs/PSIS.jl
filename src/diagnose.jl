"""
    ParetoDiagnostics

Pareto tail diagnostics returned by [`pareto_diagnose`](@ref) and
[`pareto_smooth`](@ref).

For vector or matrix observations, each field is a scalar. For batched
observations, `khat`, `min_ss`, and `convergence_rate` have the parameter shape;
`khat_threshold` is a scalar based on the pooled draw count. Each field links
to its diagnostic's interpretation below.

# Fields

$(FIELDS)
"""
struct ParetoDiagnostics{K,T<:Real,M,C}
    """
    Estimated tail shape; see [`pareto_khat`](@ref).
    """
    khat::K
    """
    Sample-size threshold; see [`pareto_khat_threshold`](@ref).
    """
    khat_threshold::T
    """
    Suggested sample size; see [`pareto_min_ss`](@ref).
    """
    min_ss::M
    """
    Relative RMSE rate; see [`pareto_convergence_rate`](@ref).
    """
    convergence_rate::C
end

function Base.:(==)(a::ParetoDiagnostics, b::ParetoDiagnostics)
    return a.khat == b.khat &&
           a.khat_threshold == b.khat_threshold &&
           a.min_ss == b.min_ss &&
           a.convergence_rate == b.convergence_rate
end

"""
    pareto_khat(x; r_eff, is_log_scale, tail, tail_length, warn) -> khat

Estimate Pareto k-hat from the selected tail(s) of observations `x`.

`x` consists of draws (or potentially log-draws) with shape
`(draws, [chains, [params...]])`.
Vectors and matrices return one value; additional dimensions index parameters.
The observations are not modified.

# Interpretation

K-hat estimates generalized Pareto tail shape; larger values indicate heavier
tails. It describes the variable on the scale of the expectation being estimated.
For an ideal Pareto tail with positive shape `k`, moments of order `p < 1/k`
exist: `k < 0.5` permits finite variance, while `k < 1` permits a finite mean.
An estimated k-hat is a finite-sample diagnostic, not proof that population
moments exist. It complements R-hat and rank-based ESS, which can look good
despite poorly behaved moments.

For interpretation in a Pareto-smoothed estimate, see
[`pareto_khat_threshold`](@ref), [`pareto_min_ss`](@ref), and
[`pareto_convergence_rate`](@ref). See also Section 3 of
[VehtariSimpson2021](@citet).

# Keywords

  - `r_eff=1`: a positive finite relative efficiency, supplied as a scalar or an array matching
    the parameter dimensions. The default assumes independent draws; for MCMC draws, an
    estimate based on tail ESS may be supplied explicitly.
  - `is_log_scale=false`: indicates whether `x` encodes positive observations on the log
    scale: the fit describes the original-scale observations, not their logarithms. If true,
    only `tail=:right` is supported.
  - `tail`: which tail(s) to fit (`:left`, `:right`, or `:both`). If `:both`, the larger of
    the k-hats of the left and right fits is returned. Default is `:both` if
    `is_log_scale=false` and `:right` if `is_log_scale=true`.
  - `tail_length`: the number of observations in the tail to use for fitting. In most cases
    this should be left as `nothing` to use a sensible default.
  - `warn=false`: whether to display warnings for failed or unreliable diagnostics.

Failed fits return `NaN`. Successful fits return k-hat even when it exceeds
the reliability threshold; `warn=true` additionally reports this condition.

To compute additional diagnostics along with the k-hat estimate, use
[`pareto_diagnose`](@ref).
"""
function pareto_khat(
    x::AbstractArray{<:Real};
    r_eff=1,
    is_log_scale::Bool=false,
    tail::Symbol=(is_log_scale ? :right : :both),
    tail_length::Union{Nothing,Integer}=nothing,
    warn::Bool=false,
)
    _check_pareto_inputs(x, r_eff, tail, is_log_scale, tail_length)
    k = _map_parameters(x, r_eff) do values, reff
        m = isnothing(tail_length) ? _tail_length(length(values), reff) : tail_length
        return _diagnose_tails(vec(values), m, tail, is_log_scale)
    end
    warn && _warn_diagnostics(k, _sample_size(x))
    return k
end

function _diagnose_tails(x, m, tail, is_log_scale)
    if tail === :both
        left = _fit_tail(x, m; tail=:left, is_log_scale)
        right = _fit_tail(x, m; tail=:right, is_log_scale)
        return max(left.dist.k, right.dist.k)
    end
    return _fit_tail(x, m; tail, is_log_scale).dist.k
end

"""
    pareto_diagnose(x; kwargs...) -> ParetoDiagnostics

Compute all Pareto diagnostics, fitting each requested tail once per variable.
Prefer this to calling several sample-based diagnostic functions separately.

For a description of the arguments, see [`pareto_khat`](@ref).
See [`ParetoDiagnostics`](@ref) for the result fields and links to their interpretation.
"""
function pareto_diagnose(x::AbstractArray{<:Real}; kwargs...)
    return _pareto_diagnostics(pareto_khat(x; kwargs...), _sample_size(x))
end

function _pareto_diagnostics(k, n)
    T = eltype(k)
    threshold = n > 1 ? T(pareto_khat_threshold(n)) : T(NaN)
    rate = n > 1 ? pareto_convergence_rate.(k, n) : oftype.(k, NaN)
    return ParetoDiagnostics(k, threshold, pareto_min_ss.(k), rate)
end

function pareto_min_ss(x::AbstractArray{<:Real}; kwargs...)
    return pareto_min_ss.(pareto_khat(x; kwargs...))
end

function pareto_khat_threshold(x::AbstractArray{<:Real})
    n = _sample_size(x)
    return n > 1 ? pareto_khat_threshold(n) : NaN
end

function pareto_convergence_rate(x::AbstractArray{<:Real}; kwargs...)
    return pareto_diagnose(x; kwargs...).convergence_rate
end

_sample_size(x) = prod(Base.Fix1(size, x), _sample_dims(x))

function _check_pareto_inputs(x, r_eff, tail, is_log_scale, tail_length)
    ndims(x) >= 1 || throw(ArgumentError("observations must have at least one dimension"))
    tail in (:left, :right, :both) ||
        throw(ArgumentError("tail must be :left, :right, or :both"))
    is_log_scale &&
        tail !== :right &&
        throw(ArgumentError("log-scale data require tail=:right"))
    all(r -> isfinite(r) && r > 0, r_eff) ||
        throw(ArgumentError("r_eff must be positive and finite"))
    shape = map(length, _param_axes(x))
    (length(r_eff) == 1 || size(r_eff) == shape) ||
        throw(DimensionMismatch("r_eff must be a scalar or match parameter shape $shape"))
    if !isnothing(tail_length)
        n = _sample_size(x)
        limit = tail === :both ? (n - 1) ÷ 2 : n - 1
        5 <= tail_length <= limit ||
            throw(ArgumentError("tail_length must be between 5 and $limit"))
    end
    return nothing
end

function _map_parameters(f, x::AbstractVecOrMat, r_eff)
    return f(x, first(r_eff))
end
function _map_parameters(f, x::AbstractArray, r_eff)
    result = similar(x, float(eltype(x)), _param_axes(x))
    for i in _eachparamindex(x)
        reff = length(r_eff) == 1 ? first(r_eff) : r_eff[i]
        result[i] = f(_selectparam(x, i), reff)
    end
    return result
end

function _warn_diagnostics(k, n)
    nfailed = count(isnan, k)
    nfailed > 0 &&
        @warn "For $nfailed variables, the generalized Pareto distribution could not be fit to the tail draws."
    threshold = _psis_threshold(n)
    nbad = count(>(threshold), k)
    nbad > 0 &&
        @warn "$nbad variables have Pareto k-hat above $threshold; Pareto-smoothed estimates may be unstable or have high bias."
    return nothing
end

_psis_threshold(n) = n > 1 ? min(0.7, pareto_khat_threshold(n)) : NaN
