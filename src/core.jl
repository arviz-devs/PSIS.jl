"""
    PSISResult

Result of Pareto-smoothed importance sampling (PSIS) using [`psis`](@ref).

# Properties

  - `log_weights`: un-normalized Pareto-smoothed log weights
  - `weights`: normalized Pareto-smoothed weights (allocates a copy)
  - `pareto_shape`: Pareto ``k=ξ`` shape parameter
  - `nparams`: number of parameters in `log_weights`
  - `ndraws`: number of draws in `log_weights`
  - `nchains`: number of chains in `log_weights`
  - `reff`: the ratio of the effective sample size of the unsmoothed importance ratios and
    the actual sample size.
  - `ess`: estimated effective sample size of estimate of mean using smoothed importance
    samples (see [`ess_is`](@ref))
  - `tail_length`: length of the upper tail of `log_weights` that was smoothed
  - `tail_dist`: the generalized Pareto distribution that was fit to the tail of
    `log_weights`. Note that the tail weights are scaled to have a maximum of 1, so
    `tail_dist * exp(maximum(log_ratios))` is the corresponding fit directly to the tail of
    `log_ratios`.
  - `normalized::Bool`:indicates whether `log_weights` are log-normalized along the sample
    dimensions.

# Diagnostic

The `pareto_shape` parameter ``k=ξ`` of the generalized Pareto distribution `tail_dist` can
be used to diagnose reliability and convergence of estimates using the importance weights
[^VehtariSimpson2021].

  - if ``k < \\frac{1}{3}``, importance sampling is stable, and importance sampling (IS) and
    PSIS both are reliable.
  - if ``k ≤ \\frac{1}{2}``, then the importance ratio distributon has finite variance, and
    the central limit theorem holds. As ``k`` approaches the upper bound, IS becomes less
    reliable, while PSIS still works well but with a higher RMSE.
  - if ``\\frac{1}{2} < k ≤ 0.7``, then the variance is infinite, and IS can behave quite
    poorly. However, PSIS works well in this regime.
  - if ``0.7 < k ≤ 1``, then it quickly becomes impractical to collect enough importance
    weights to reliably compute estimates, and importance sampling is not recommended.
  - if ``k > 1``, then neither the variance nor the mean of the raw importance ratios
    exists. The convergence rate is close to zero, and bias can be large with practical
    sample sizes.

See [`PSISPlots.paretoshapeplot`](@ref) for a diagnostic plot.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
struct PSISResult{T,W<:AbstractArray{T},R,D}
    log_weights::W
    reff::R
    normalized::Bool
    diagnostics::D
end

check_pareto_diagnostics(r::PSISResult) = check_pareto_diagnostics(r.diagnostics)

function Base.propertynames(r::PSISResult)
    return [fieldnames(typeof(r))..., :weights, :nparams, :ndraws, :nchains, :pareto_shape]
end

function Base.getproperty(r::PSISResult, k::Symbol)
    if k === :weights
        log_weights = getfield(r, :log_weights)
        getfield(r, :normalized) && return exp.(log_weights)
        return LogExpFunctions.softmax(log_weights; dims=_sample_dims(log_weights))
    elseif k === :nparams
        log_weights = getfield(r, :log_weights)
        return if ndims(log_weights) == 1
            1
        else
            param_dims = _param_dims(log_weights)
            prod(Base.Fix1(size, log_weights), param_dims; init=1)
        end
    elseif k === :ndraws
        log_weights = getfield(r, :log_weights)
        return size(log_weights, 1)
    elseif k === :nchains
        log_weights = getfield(r, :log_weights)
        return size(log_weights, 2)
    end
    k === :pareto_shape && return pareto_shape(getfield(r, :diagnostics))
    k === :ess && return ess_is(r)
    return getfield(r, k)
end

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    npoints = r.nparams
    nchains = r.nchains
    println(
        io, "PSISResult with $(r.ndraws) draws, $nchains chains, and $npoints parameters"
    )
    return _print_pareto_shape_summary(io, r; newline_at_end=false)
end

function pareto_shape_summary(r::PSISResult; kwargs...)
    return _print_pareto_shape_summary(stdout, r; kwargs...)
end

function _print_pareto_shape_summary(io::IO, r::PSISResult; kwargs...)
    k = as_array(pareto_shape(r))
    sample_size = r.ndraws * r.nchains
    ess = as_array(ess_is(r))
    diag = _compute_diagnostics(k, sample_size)

    category_assignments = NamedTuple{(:good, :bad, :very_bad, :failed)}(
        _diagnostic_category_assignments(diag)
    )
    category_intervals = _diagnostic_intervals(diag)
    npoints = length(k)
    rows = map(collect(pairs(category_assignments))) do (desc, inds)
        interval = desc === :failed ? "--" : _interval_string(category_intervals[desc])
        min_ess = @views isempty(inds) ? NaN : minimum(ess[inds])
        return (; interval, desc, count=length(inds), min_ess)
    end
    _print_pareto_diagnostics_summary(io::IO, rows, npoints; kwargs...)
    return nothing
end

pareto_shape(r::PSISResult) = pareto_shape(r.diagnostics)

"""
    psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

# Arguments

  - `log_ratios`: an array of logarithms of importance ratios, with size
    `(draws, [chains, [parameters...]])`, where `chains>1` would be used when chains are
    generated using Markov chain Monte Carlo.
  - `reff::Union{Real,AbstractArray}`: the ratio(s) of effective sample size of
    `log_ratios` and the actual sample size `reff = ess/(draws * chains)`, used to account
    for autocorrelation, e.g. due to Markov chain Monte Carlo. If an array, it must have the
    size `(parameters...,)` to match `log_ratios`.

# Keywords

  - `warn=true`: If `true`, warning messages are delivered
  - `normalize=true`: If `true`, the log-weights will be log-normalized so that
    `exp.(log_weights)` sums to 1 along the sample dimensions.

# Returns

  - `result`: a [`PSISResult`](@ref) object containing the results of the Pareto-smoothing.

A warning is raised if the Pareto shape parameter ``k ≥ 0.7``. See [`PSISResult`](@ref) for
details and [`PSISPlots.paretoshapeplot`](@ref) for a diagnostic plot.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
psis

function psis(logr::AbstractArray{<:Real}; normalize::Bool=true, reff=1, kwargs...)
    logw, diagnostics = pareto_smooth(logr; is_log=true, tails=RightTail, reff, kwargs...)
    if normalize
        logw .-= LogExpFunctions.logsumexp(logw; dims=_sample_dims(logw))
    end
    return PSISResult(logw, reff, normalize, diagnostics)
end
