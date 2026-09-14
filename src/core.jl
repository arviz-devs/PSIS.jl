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

Use [`pareto_diagnose`](@ref) on a result to obtain additional diagnostics without
refitting. See [VehtariSimpson2021](@citep).
"""
struct PSISResult{T,W<:AbstractArray{T},K,E}
    """
    Un-normalized Pareto-smoothed log importance weights.
    """
    log_weights::W
    """
    Pareto k-hat (shape) diagnostic.
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
    return _print_pareto_khat_summary(io, r; newline_at_end=false)
end

function _print_pareto_khat_summary(io::IO, r::PSISResult; kwargs...)
    k = as_array(pareto_khat(r))
    ess_is = as_array(r.ess_is)
    (_, _, nparams) = _sample_param_sizes(r.log_weights)
    if isempty(k)
        print(io, "No parameter diagnostics.")
        return nothing
    end
    threshold = _psis_threshold(_sample_size(r.log_weights))
    label = @sprintf("%.3g", threshold)
    categories = (
        ("(-Inf, $label]", "good", <=(threshold)),
        ("($label, 1]", "bad", k -> threshold < k <= 1),
        ("(1, Inf)", "very bad", >(1)),
        ("——", "failed", isnan),
    )
    rows = map(categories) do (range, desc, cond)
        inds = findall(cond, k)
        count = length(inds)
        perc = 100 * count / nparams
        ess_min = if count == 0 || desc == "failed"
            oftype(first(ess_is), NaN)
        else
            minimum(view(ess_is, inds))
        end
        return (range=range, desc=desc, count_perc=(count, perc), ess_min=ess_min)
    end
    rows = filter(r -> r.count_perc[1] > 0, rows)
    formats = Dict(
        "good" => (),
        "bad" => (bold=true, color=:light_red),
        "very bad" => (bold=true, color=:red),
        "failed" => (; color=:red),
    )

    col_padding = " "
    col_delim = ""
    col_delim_tot = col_padding * col_delim * col_padding
    col_widths = [
        maximum(r -> length(r.range), rows),
        maximum(r -> length(r.desc), rows),
        maximum(r -> ndigits(r.count_perc[1]), rows),
        floor(Int, log10(maximum(r -> r.count_perc[2], rows))) + 6,
    ]

    println(io, "Pareto k-hat diagnostic summary:")
    printstyled(
        io,
        col_padding,
        " "^col_widths[1],
        col_delim_tot,
        " "^col_widths[2],
        col_delim_tot,
        _pad_right("Count", col_widths[3] + col_widths[4] + 1),
        col_delim_tot,
        "Min. ESS";
        bold=true,
    )
    for r in rows
        count, perc = r.count_perc
        perc_str = "($(round(perc; digits=1))%)"
        println(io)
        print(io, col_padding, _pad_left(r.range, col_widths[1]), col_delim_tot)
        print(io, _pad_right(r.desc, col_widths[2]), col_delim_tot)
        format = formats[r.desc]
        printstyled(io, _pad_left(count, col_widths[3]); format...)
        printstyled(io, " ", _pad_right(perc_str, col_widths[4]); format...)
        ess_min_string = if isfinite(r.ess_min) && r.desc == "good"
            string(floor(Int, r.ess_min))
        else
            "——"
        end
        print(io, col_delim_tot, ess_min_string)
    end
    return nothing
end

_pad_left(s, nchars) = " "^(nchars - length("$s")) * "$s"
_pad_right(s, nchars) = "$s" * " "^(nchars - length("$s"))

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
