"""
    pareto_khat(x; r_eff=1, tail=:both, is_log_scale=false, tail_length=nothing, warn=false)

Estimate Pareto k-hat from the selected tail(s) of observations `x`, without
modifying them. For `tail=:both`, return the larger of the left and right fits.
Vectors and matrices represent one variable, with dimensions `(draws, chains)`;
higher-dimensional arrays return one value per remaining parameter index.

`r_eff` is a positive finite relative efficiency, supplied as a scalar or an
array matching the parameter dimensions. The default assumes independent draws;
for MCMC draws, an estimate based on tail ESS may be supplied explicitly.
The default tail length is `floor(min(n/5, 3sqrt(n/r_eff)))`.

`is_log_scale=true` means `x` encodes positive observations on the log scale:
the fit describes the original-scale observations, not their logarithms. Only
the right tail is supported in this mode, and it becomes the default.

Constant or nonfinite tails and tails with fewer than five observations return
`NaN`. Negative infinity in log-encoded data denotes zero and is allowed outside
the fitted tail. `warn=true` reports failed or unreliable diagnostics.
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
    pareto_diagnose(x; kwargs...)

Return `(khat, khat_threshold, min_ss, convergence_rate)` as a named tuple.
Keywords are those of [`pareto_khat`](@ref). Each variable is fitted once.
The threshold uses the pooled number of draws, while the remaining fields have
one value per variable. Scalar and named-tuple outputs can be used as summary
functions, including with `PosteriorStats.summarize`.

These diagnose tail behavior; they do not replace MCMC convergence diagnostics.
The minimum sample size and convergence rate describe a Pareto-smoothed estimate.
"""
function pareto_diagnose(x::AbstractArray{<:Real}; kwargs...)
    return _pareto_diagnostics(pareto_khat(x; kwargs...), _sample_size(x))
end

function _pareto_diagnostics(k, n)
    T = eltype(k)
    threshold = n > 1 ? T(ps_khat_threshold(n)) : T(NaN)
    rate = n > 1 ? ps_convergence_rate.(k, n) : oftype.(k, NaN)
    return (; khat=k, khat_threshold=threshold, min_ss=ps_min_ss.(k), convergence_rate=rate)
end

"""
    pareto_min_ss(x; kwargs...)

Estimate the minimum sample size for a Pareto-smoothed estimate of `x`.
Accepts the keywords of [`pareto_khat`](@ref). See [`ps_min_ss`](@ref) for
the calculation from an existing k-hat.
"""
pareto_min_ss(x::AbstractArray{<:Real}; kwargs...) = ps_min_ss.(pareto_khat(x; kwargs...))

"""
    pareto_khat_threshold(x)

Return the sample-size-dependent threshold for observations `x`, pooling the
first two dimensions. Returns `NaN` for fewer than two draws.
See [`ps_khat_threshold`](@ref).
"""
function pareto_khat_threshold(x::AbstractArray{<:Real})
    n = _sample_size(x)
    return n > 1 ? ps_khat_threshold(n) : NaN
end

"""
    pareto_convergence_rate(x; kwargs...)

Estimate the relative RMSE convergence rate of a Pareto-smoothed estimate of `x`.
Accepts the keywords of [`pareto_khat`](@ref). See [`ps_convergence_rate`](@ref)
for its interpretation and calculation from an existing k-hat.
"""
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

_psis_threshold(n) = n > 1 ? min(0.7, ps_khat_threshold(n)) : NaN
