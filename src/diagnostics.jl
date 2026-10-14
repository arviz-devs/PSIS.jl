"""
    ParetoDiagnostics

Pareto diagnostics returned by [`pareto_diagnose`](@ref)

# Fields

$(FIELDS)
"""
struct ParetoDiagnostics{T<:Real,V<:Union{T,AbstractArray{T}}}
    """
    Estimated Pareto k shape parameter; see [`pareto_khat`](@ref).
    """
    khat::V
    """
    khat-threshold for reliable Pareto smoothed estimate; see [`pareto_khat_threshold`](@ref).
    """
    khat_threshold::T
    """
    Minimum sample size for reliable Pareto smoothed estimate; see [`pareto_min_ss`](@ref).
    """
    min_ss::V
    """
    Relative RMSE rate for the Pareto smoothed estimate; see [`pareto_convergence_rate`](@ref).
    """
    convergence_rate::V
end

function Base.show(io::IO, ::MIME"text/plain", diagnostics::ParetoDiagnostics)
    nparams = length(as_array(diagnostics.khat))
    parameter_label = nparams == 1 ? "parameter" : "parameters"
    println(io, "ParetoDiagnostics with $nparams $parameter_label")
    return show(io, MIME"text/plain"(), pareto_k_table(diagnostics))
end

function pareto_k_table(diagnostics::ParetoDiagnostics)
    return pareto_k_table(diagnostics.khat, min(0.7, diagnostics.khat_threshold))
end

"""
    pareto_diagnose(x; kwargs...) -> ParetoDiagnostics

Compute all Pareto diagnostics for draws of one or more variables.

See [`pareto_khat`](@ref) for a description of `x` and the keyword arguments,
and [`ParetoDiagnostics`](@ref) for a description of the diagnostics.
"""
function pareto_diagnose(x::AbstractArray{<:Real}; kwargs...)
    return _pareto_diagnostics(pareto_khat(x; kwargs...), _sample_size(x))
end

function _pareto_diagnostics(k, n)
    T = eltype(k)
    threshold = T(pareto_khat_threshold(n))
    min_ss = T.(pareto_min_ss.(k))
    rate = T.(pareto_convergence_rate.(k, n))
    return ParetoDiagnostics(k, threshold, min_ss, rate)
end

"""
    pareto_khat(x; kwargs...) -> khat

Estimate Pareto k-hat by fitting a Generalized Pareto distribution to the
selected tail(s) of observations `x`.

`x` contains draws (or log-draws) with shape `(draws, [chains, [params...]])`.
`khat` has shape `params` and is a scalar if `x` is a vector or matrix.
Failed fits for a given variable return `NaN`.

To compute additional diagnostics along with the k-hat estimate, use
[`pareto_diagnose`](@ref).

# Interpretation

`inv(khat)` is an estimate of the number of fractional moments that exist for
the sampling distribution. [VehtariSimpson2021](@citet) gives a number of
theoretical and empirical guidelines for interpreting k-hat values, which
are summarized here for sample size `n = draws * chains`:

  - `khat < 0.5` and `n >= 100`: the mean and variance of the target distribution
    are expected to exist and be reliable, so the Monte Carlo standard error (MCSE)
    of the mean is expected to be reliable even without Pareto smoothing.
  - `khat < min(pareto_khat_threshold(n), 0.7)`: The Pareto-smoothed estimate and the
    corresponding MCSE estimate are expected to be reliable.
  - `pareto_khat_threshold(n) <= khat < 0.7`: the Pareto-smoothed estimate and the
    corresponding MCSE estimate are not reliable, but increasing the (effective)
    sample size above 2200 may help.
  - `0.7 <= khat < 1`: The Pareto-smoothed estimate and the corresponding MCSE have
    large bias and are not reliable.
  - `khat >= 1`: The target distribution is estimated to have a non-finite mean.
    Pareto-smoothed estimates and the corresponding MCSE are not well defined.

# Keywords

  - `r_eff=1`: relative effective sample size estimate used to compute tail length
    if not provided; for Markov Chain Monte Carlo (MCMC) draws, an estimate based on
    relative tail ESS on linear scale should be provided
    (e.g. [`ess(x; kind=:tail, relative=true)`](@extref MCMCDiagnosticTools.ess)).
  - `is_log_scale=false`: indicates whether `x` encodes positive observations on the
    log scale. If `true`, only `tail=:right` is supported, and the tail distribution
    is fit on linear scale observations, correctly avoiding under/overflow.
  - `tail`: which tail(s) to fit (`:left`, `:right`, or `:both`). If `:both`, the
    larger of the k-hats of the left and right fits is returned. Default is `:both` if
    `is_log_scale=false` and `:right` if `is_log_scale=true`.
  - `tail_length`: the number of observations in the tail to use for fitting. In most
    cases this should be left as `nothing` to use a sensible default.
  - `warn=false`: whether to display warnings for failed or unreliable diagnostics.
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
        return _pareto_khat_tails(vec(values), m, tail, is_log_scale)
    end
    warn && _warn_diagnostics(k, _sample_size(x))
    return k
end

function _pareto_khat_tails(x, m, tail, is_log_scale)
    if tail === :both
        left = _fit_tail(x, m; tail=:left, is_log_scale)
        right = _fit_tail(x, m; tail=:right, is_log_scale)
        return max(left.dist.k, right.dist.k)
    end
    return _fit_tail(x, m; tail, is_log_scale).dist.k
end

"""
    pareto_min_ss(k::Real)

Compute the minimum sample size for a reliable Pareto smoothed estimate
from Pareto shape `k`.

# Interpretation

If the actual sample size is *greater* than `min_ss`, then Pareto-smoothed
estimates are unlikely to have large errors. Otherwise, increasing the
sample size may result in more reliable estimates.

See Section 3.2.3, Equation 11 in [VehtariSimpson2021](@citet).
"""
function pareto_min_ss(k::Real)
    T = float(typeof(k))
    k > 1 && return T(Inf)
    return exp10(inv(1 - max(zero(T), k)))
end

"""
    pareto_min_ss(x::AbstractArray; kwargs...)

Compute the minimum sample size for a reliable Pareto-smoothed estimate from
observations `x`.

See [`pareto_khat`](@ref) for a description of `x` and the keyword arguments.
"""
function pareto_min_ss(x::AbstractArray{<:Real}; kwargs...)
    return pareto_min_ss.(pareto_khat(x; kwargs...))
end

"""
    pareto_khat_threshold(n::Real)

Compute the k-hat threshold required for sample with size `n` to have reliable
Pareto-smoothed estimates.

# Interpretation

K-hat values *below* `khat_threshold` indicate that Pareto-smoothed
estimates are unlikely to have large errors. Otherwise, increasing the total
sample size may result in more reliable estimates.

Note that k-hat values must additionally be less than 0.7 for Pareto-smoothed
estimates and their MCSE to have low bias.

See Section 3.2.4, Equation 13 in [VehtariSimpson2021](@citet).
"""
function pareto_khat_threshold(n::Real)
    T = typeof(inv(log(oneunit(n))))
    n < 10 && return T(0)
    return 1 - inv(log10(n))
end

"""
    pareto_khat_threshold(x::AbstractArray)

Compute the k-hat threshold from the number of observations in `x`, pooling
the draws and chains dimensions.

See [`pareto_khat`](@ref) for a description of the layout of `x`.
"""
pareto_khat_threshold(x::AbstractArray{<:Real}) = pareto_khat_threshold(_sample_size(x))

"""
    pareto_convergence_rate(k::Real, n::Real)

Compute the relative RMSE convergence rate of a Pareto-smoothed estimate from
Pareto shape `k` and sample size `n`.

# Interpretation

This is the relative convergence rate compared with the central limit theorem
(i.e. relative efficiency) and applies only when the actual sample size is larger
than [`pareto_min_ss`](@ref)`(k)`.

See Appendix B in [VehtariSimpson2021](@citet) for further details.
"""
function pareto_convergence_rate(k::Real, n::Real)
    T = float(promote_type(typeof(k), typeof(n)))
    n <= 1 && return T(NaN)
    k <= 0 && return one(T)
    k >= 1 && return zero(T)

    logn = log(T(n))
    a = 2 * T(k) - 1
    z = a * logn
    # a / expm1(z) → 1/log(n) as a → 0
    ratio = iszero(z) ? inv(logn) : a / expm1(z)
    rate = 1 + inv(T(n) - 1) - a - ratio
    return clamp(rate, 0, 1)
end

"""
    pareto_convergence_rate(x::AbstractArray; kwargs...)

Compute the relative RMSE convergence rate of a Pareto-smoothed estimate from
observations `x`.

See [`pareto_khat`](@ref) for a description of `x` and the keyword arguments.
"""
function pareto_convergence_rate(x::AbstractArray{<:Real}; kwargs...)
    n = _sample_size(x)
    k = pareto_khat(x; kwargs...)
    return n > 1 ? pareto_convergence_rate.(k, n) : oftype.(k, NaN)
end

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

_map_parameters(f, x::AbstractVecOrMat, r_eff) = f(x, first(r_eff))
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
    threshold = _low_bias_mcse_threshold(n)
    nbad = count(>(threshold), k)
    nbad > 0 &&
        @warn "$nbad variables have Pareto k-hat above $threshold; Pareto-smoothed estimates may be unstable or have high bias."
    return nothing
end

_low_bias_mcse_threshold(n) = min(0.7, pareto_khat_threshold(n))
