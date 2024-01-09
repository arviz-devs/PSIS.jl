"""
    PSISResult

Result of Pareto-smoothed importance sampling (PSIS) using [`psis`](@ref).

$FIELDS

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
struct PSISResult{T,W<:AbstractArray{T},R,D<:ParetoDiagnostics}
    "Pareto-smoothed log-weights. Log-normalized if `normalized=true`."
    log_weights::W
    "the relative efficiency, i.e. the ratio of the effective sample size of the unsmoothed
    importance ratios and the actual sample size."
    reff::R
    "whether `log_weights` are log-normalized along the sample dimensions."
    normalized::Bool
    "diagnostics for the Pareto-smoothing."
    diagnostics::D
end

check_pareto_diagnostics(r::PSISResult) = check_pareto_diagnostics(r.diagnostics)

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    log_weights = r.log_weights
    ndraws = size(log_weights, 1)
    nchains = size(log_weights, 2)
    npoints = prod(_param_sizes(log_weights))
    println(
        io, "PSISResult with $ndraws draws, $nchains chains, and $npoints parameters"
    )
    return _print_pareto_shape_summary(io, r; newline_at_end=false)
end

function _print_pareto_shape_summary(io::IO, r::PSISResult; kwargs...)
    k = as_array(pareto_shape(r))
    sample_size = _sample_size(r.log_weights)
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
