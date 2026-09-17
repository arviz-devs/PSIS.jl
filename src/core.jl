"""
    PSISResult

Result of Pareto-smoothed importance sampling (PSIS) using [`psis`](@ref).

# Fields

$(FIELDS)

# Diagnostic

The fitted shape diagnoses the raw importance ratios. A result is considered
reliable when `k <= min(0.7, 1 - 1/log10(n))`, where `n` pools draws and chains.
The `ess_is` field is `NaN` for unreliable or failed fits. This does not establish
reliability for every possible expectand.

See [`pareto_khat`](@ref) and [`pareto_khat_threshold`](@ref) for interpretation.

Use [`pareto_diagnose`](@ref) on a result to obtain additional diagnostics without
refitting. See [VehtariSimpson2021](@citep).
"""
struct PSISResult{T,W<:AbstractArray{T},K,E}
    """
    Un-normalized Pareto-smoothed log importance weights.
    """
    log_weights::W
    """
    Tail shape of the raw importance ratios; see [`pareto_khat`](@ref).
    """
    pareto_khat::K
    """
    Estimated effective sample size; NaN for unreliable or failed fits.
    """
    ess_is::E
end

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    (ndraws, nchains, nparams) = _sample_param_sizes(r.log_weights)
    println(io, "PSISResult with $ndraws draws, $nchains chains, and $nparams parameters")
    threshold = _psis_threshold(_sample_size(r.log_weights))
    return _print_pareto_khat_summary(io, r.pareto_khat, threshold; ess_is=r.ess_is)
end

"""
    psis(log_ratios, r_eff=1; warn=true, tail_length=nothing) -> PSISResult

Compute unnormalized Pareto-smoothed log importance weights. The array shape is
`(draws[, chains[, parameters...]])`. Input values are never modified.

`r_eff` must be positive and finite, either a scalar or an array matching the
parameter dimensions. It controls tail length and adjusts the generic weight
ESS. The default assumes independent draws. For LOO, a common convention is the
relative efficiency of the inverse ratios; select an appropriate efficiency
estimate for the sampling procedure.

`tail_length` optionally specifies the number of right-tail observations.
Otherwise use `floor(min(n/5, 3sqrt(n/r_eff)))`, pooling draws and chains.
Weights are capped at the largest observed raw ratio, as in Algorithm 1 of
[VehtariSimpson2021](@citet).

A warning is emitted when a tail cannot be fitted or its k-hat exceeds
`min(0.7, 1 - 1/log10(n))`. Set `warn=false` to silence these warnings.
Unreliable or failed fits have `NaN` ESS. See [`PSISResult`](@ref).

# Examples

```jldoctest
julia> log_ratios = collect(range(-1.0, 0.0; length=1000));

julia> result = psis(log_ratios; warn=false);

julia> size(result.log_weights)
(1000,)

julia> maximum(result.log_weights) <= maximum(log_ratios)
true

julia> pareto_diagnose(result).khat == result.pareto_khat
true
```

For MCMC-based LOO, relative efficiency can be computed explicitly using
[`MCMCDiagnosticTools.ess`](@extref):

```julia
using LogExpFunctions, MCMCDiagnosticTools
r_eff = ess(softmax(-log_ratios; dims=(1, 2)); kind=:basic, split_chains=1, relative=true)
result = psis(log_ratios, r_eff)
```

Normalize only when needed, for example with
`softmax(result.log_weights; dims=(1, 2))` for matrix or batched draws.
"""
function psis(logr::AbstractArray{<:Real}, r_eff=1; warn::Bool=true, tail_length=nothing)
    logw, diagnostics = pareto_smooth(
        logr; r_eff, is_log_scale=true, tail=:right, tail_length, warn
    )
    k = diagnostics.khat
    threshold = _psis_threshold(_sample_size(logr))
    ess = ess_is(logw, r_eff)
    reliable_ess = ifelse.(isfinite.(k) .& (k .<= threshold), ess, oftype.(ess, NaN))
    return PSISResult(logw, k, reliable_ess)
end

pareto_khat(dist::GeneralizedPareto) = dist.k
pareto_khat(r::PSISResult) = r.pareto_khat

"""
    pareto_diagnose(result::PSISResult)

Return diagnostics from the saved raw-ratio k-hat and sample size, without
refitting the smoothed weights.
"""
function pareto_diagnose(r::PSISResult)
    return _pareto_diagnostics(r.pareto_khat, _sample_size(r.log_weights))
end
