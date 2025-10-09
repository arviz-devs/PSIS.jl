# range, description, condition
const SHAPE_DIAGNOSTIC_CATEGORIES = (
    ("(-Inf, 0.5]", "good", ≤(0.5)),
    ("(0.5, 0.7]", "okay", x -> 0.5 < x ≤ 0.7),
    ("(0.7, 1]", "bad", x -> 0.7 < x ≤ 1),
    ("(1, Inf)", "very bad", >(1)),
    ("——", "failed", isnan),
)
const BAD_SHAPE_SUMMARY = "Resulting importance sampling estimates are likely to be unstable."
const VERY_BAD_SHAPE_SUMMARY = "Corresponding importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples."

"""
    PSISResult

Result of Pareto-smoothed importance sampling (PSIS) using [`psis`](@ref).

# Fields

$FIELDS

# Diagnostic

The `pareto_shape` diagnostic ``k`` of the generalized Pareto distribution can be used to
diagnose reliability and convergence of estimates using the importance weights
[VehtariSimpson2021](@citep).

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

# References

  - [VehtariSimpson2021](@cite) Vehtari et al. JMLR 25:72 (2021).
"""
struct PSISResult{T<:Real,L<:AbstractArray{T},P<:Union{T,AbstractArray{T}}}
    """Un-normalized Pareto-smoothed log weights with shape
    `(draws, [chains, [parameters...]])`"""
    log_weights::L
    """Pareto shape ``k`` diagnostic values for each parameter"""
    pareto_shape::P
    """Relative efficiency of each parameter, the ratio of the effective sample size of
    the unsmoothed importance ratios and the actual sample size."""
    reff::P
end

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    ndraws = _ndraws(r)
    nchains = _nchains(r)
    npoints = _npoints(r)
    println(io, "PSISResult with $ndraws draws, $nchains chains, and $npoints parameters")
    return _print_pareto_shape_summary(io, r; newline_at_end=false)
end

_ndraws(r::PSISResult) = size(r.log_weights, 1)
_nchains(r::PSISResult) = size(r.log_weights, 2)
_npoints(r::PSISResult) = prod(size(r.log_weights)[3:end])

function pareto_shape_summary(r::PSISResult; kwargs...)
    return _print_pareto_shape_summary(stdout, r; kwargs...)
end

function _print_pareto_shape_summary(io::IO, r::PSISResult; kwargs...)
    k = as_array(r.pareto_shape)
    ess = as_array(ess_is(r))
    npoints = _npoints(r)
    rows = map(SHAPE_DIAGNOSTIC_CATEGORIES) do (range, desc, cond)
        inds = findall(cond, k)
        count = length(inds)
        perc = 100 * count / npoints
        ess_min = if count == 0 || desc == "failed"
            oftype(first(ess), NaN)
        else
            minimum(view(ess, inds))
        end
        return (range=range, desc=desc, count_perc=(count, perc), ess_min=ess_min)
    end
    rows = filter(r -> r.count_perc[1] > 0, rows)
    formats = Dict(
        "good" => (),
        "okay" => (; color=:yellow),
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

    println(io, "Pareto shape (k) diagnostic values:")
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
        print(io, col_delim_tot, isfinite(r.ess_min) ? floor(Int, r.ess_min) : "——")
    end
    return nothing
end

_pad_left(s, nchars) = " "^(nchars - length("$s")) * "$s"
_pad_right(s, nchars) = "$s" * " "^(nchars - length("$s"))

"""
    psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult
    psis!(log_ratios, reff = 1.0; kwargs...) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [VehtariSimpson2021](@citep).

While `psis` computes smoothed log weights out-of-place, `psis!` smooths them in-place.

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

# Examples

Here we smooth log importance ratios for importance sampling 30 isotropic Student
``t``-distributed parameters using standard normal distributions as proposals.

```jldoctest psis; setup = :(using Random; Random.seed!(42))
julia> using Distributions

julia> proposal, target = Normal(), TDist(7);

julia> x = rand(proposal, 1_000, 1, 30);  # (ndraws, nchains, nparams)

julia> log_ratios = @. logpdf(target, x) - logpdf(proposal, x);

julia> result = psis(log_ratios)
┌ Warning: 9 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.
└ @ PSIS ~/.julia/packages/PSIS/...
┌ Warning: 1 parameters had Pareto shape values k > 1. Corresponding importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples.
└ @ PSIS ~/.julia/packages/PSIS/...
PSISResult with 1000 draws, 1 chains, and 30 parameters
Pareto shape (k) diagnostic values:
                        Count       Min. ESS
 (-Inf, 0.5]  good       7 (23.3%)  959
  (0.5, 0.7]  okay      13 (43.3%)  938
    (0.7, 1]  bad        9 (30.0%)  ——
    (1, Inf)  very bad   1 (3.3%)   ——
```

If the draws were generated using MCMC, we can compute the relative efficiency using
[`MCMCDiagnosticTools.ess`](@extref).

```jldoctest psis
julia> using MCMCDiagnosticTools

julia> reff = ess(log_ratios; kind=:basic, split_chains=1, relative=true);

julia> result = psis(log_ratios, reff)
┌ Warning: 9 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.
└ @ PSIS ~/.julia/packages/PSIS/...
┌ Warning: 1 parameters had Pareto shape values k > 1. Corresponding importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples.
└ @ PSIS ~/.julia/packages/PSIS/...
PSISResult with 1000 draws, 1 chains, and 30 parameters
Pareto shape (k) diagnostic values:
                        Count       Min. ESS
 (-Inf, 0.5]  good       9 (30.0%)  806
  (0.5, 0.7]  okay      11 (36.7%)  842
    (0.7, 1]  bad        9 (30.0%)  ——
    (1, Inf)  very bad   1 (3.3%)   ——
```

# References

  - [VehtariSimpson2021](@cite) Vehtari et al. JMLR 25:72 (2021).
"""
psis, psis!

function psis(logr, reff=1; kwargs...)
    T = float(eltype(logr))
    logw = similar(logr, T)
    copyto!(logw, logr)
    return psis!(logw, reff; kwargs...)
end

function psis!(logw::AbstractVector, reff=1; warn::Bool=true)
    T = typeof(float(one(eltype(logw))))
    if length(reff) != 1
        throw(DimensionMismatch("`reff` has length $(length(reff)) but must have length 1"))
    end
    warn && check_reff(reff)
    S = length(logw)
    reff_val = T(first(reff))
    M = tail_length(reff_val, S)
    if M < 5
        warn &&
            @warn "$M tail draws is insufficient to fit the generalized Pareto distribution. Total number of draws should in general exceed 25."
        return _failed_psis_result(logw, reff_val)
    end
    perm = partialsortperm(logw, (S - M):S)
    cutoff_ind = perm[1]
    tail_inds = @view perm[2:(M + 1)]
    logu = logw[cutoff_ind]
    logw_tail = @views logw[tail_inds]
    if !all(isfinite, logw_tail)
        warn &&
            @warn "Tail contains non-finite values. Generalized Pareto distribution cannot be reliably fit."
        return _failed_psis_result(logw, reff_val)
    end
    _, pareto_shape = psis_tail!(logw_tail, logu)
    warn && check_pareto_shape(pareto_shape)
    return PSISResult(logw, pareto_shape, reff_val)
end
function psis!(logw::AbstractMatrix, reff=1; kwargs...)
    result = psis!(vec(logw), reff; kwargs...)
    return PSISResult(logw, result.pareto_shape, result.reff)
end
function psis!(logw::AbstractArray, reff=1; warn::Bool=true)
    T = typeof(float(one(eltype(logw))))
    # if an array defines custom indices (e.g. AbstractDimArray), we preserve them
    param_axes = _param_axes(logw)
    param_shape = map(length, param_axes)
    if !(length(reff) == 1 || size(reff) == param_shape)
        throw(
            DimensionMismatch(
                "`reff` has shape $(size(reff)) but must have same shape as the parameter axes $(param_shape)",
            ),
        )
    end
    check_reff(reff)

    # allocate containers
    reffs = similar(logw, float(eltype(reff)), param_axes)
    reffs .= reff
    pareto_shape = similar(logw, T, param_axes)

    # call psis! in parallel for all parameters
    Threads.@threads for i in _eachparamindex(logw)
        logw_i = _selectparam(logw, i)
        result_i = psis!(logw_i, reffs[i]; warn=false)
        pareto_shape[i] = result_i.pareto_shape[1]
    end

    # combine results
    result = PSISResult(logw, pareto_shape, reffs)

    # warn for bad shape
    warn && check_pareto_shape(result)
    return result
end

function _failed_psis_result(logw::AbstractArray, reff::Number)
    T = eltype(logw)
    return PSISResult(logw, T(NaN), reff)
end

function check_reff(reff)
    isvalid = all(reff) do r
        return isfinite(r) && r > 0
    end
    isvalid || @warn "All values of `reff` should be finite, but some are not."
    return nothing
end

check_pareto_shape(result::PSISResult) = check_pareto_shape(result.pareto_shape)
function check_pareto_shape(k::Real)
    if k > 1
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 1. $VERY_BAD_SHAPE_SUMMARY"
    elseif k > 0.7
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 0.7. $BAD_SHAPE_SUMMARY"
    end
    return nothing
end
function check_pareto_shape(pareto_shapes::AbstractArray{<:Real})
    nnan = count(isnan, pareto_shapes)
    ngt07 = count(>(0.7), pareto_shapes)
    ngt1 = iszero(ngt07) ? ngt07 : count(>(1), pareto_shapes)
    if ngt07 > ngt1
        @warn "$(ngt07 - ngt1) parameters had Pareto shape values 0.7 < k ≤ 1. $BAD_SHAPE_SUMMARY"
    end
    if ngt1 > 0
        @warn "$ngt1 parameters had Pareto shape values k > 1. $VERY_BAD_SHAPE_SUMMARY"
    end
    if nnan > 0
        @warn "For $nnan parameters, the generalized Pareto distribution could not be fit to the tail draws. Total number of draws should in general exceed 25, and the tail draws must be finite."
    end
    return nothing
end

function tail_length(reff, S)
    max_length = cld(S, 5)
    (isfinite(reff) && reff > 0) || return max_length
    min_length = ceil(Int, 3 * sqrt(S / reff))
    return min(max_length, min_length)
end

function psis_tail!(logw, logμ)
    T = eltype(logw)
    logw_max = logw[end]
    # to improve numerical stability, we first shift the log-weights to have a maximum of 0,
    # equivalent to scaling the weights to have a maximum of 1.
    μ_scaled = exp(logμ - logw_max)
    w_scaled = (logw .= exp.(logw .- logw_max) .- μ_scaled)
    tail_dist = fit_gpd(w_scaled; prior_adjusted=true, sorted=true)
    # undo the scaling
    (; k) = tail_dist
    if isfinite(k)
        p = uniform_probabilities(T, length(logw))
        @inbounds for i in eachindex(logw, p)
            # undo scaling in the log-weights
            logw[i] = min(log(quantile(tail_dist, p[i]) + μ_scaled), 0) + logw_max
        end
    end
    return logw, k
end
