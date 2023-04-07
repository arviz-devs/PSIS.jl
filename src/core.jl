# range, description, condition
const SHAPE_DIAGNOSTIC_CATEGORIES = (
    ("(-Inf, 0.5]", "good", x -> !ismissing(x) && x ≤ 0.5),
    ("(0.5, 0.7]", "okay", x -> !ismissing(x) && 0.5 < x ≤ 0.7),
    ("(0.7, 1]", "bad", x -> !ismissing(x) && 0.7 < x ≤ 1),
    ("(1, Inf)", "very bad", x -> !ismissing(x) && x > 1),
    ("——", "missing", ismissing),
)
const BAD_SHAPE_SUMMARY = "Resulting importance sampling estimates are likely to be unstable."
const VERY_BAD_SHAPE_SUMMARY = "Corresponding importance sampling estimates are likely to be unstable and are unlikely to converge with additional samples."

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
  - `log_weights_norm`: the logarithm of the normalization constant of `log_weights`
  - `tail_length`: length of the upper tail of `log_weights` that was smoothed
  - `tail_dist`: the generalized Pareto distribution that was fit to the tail of
    `log_weights`. Note that the tail weights are scaled to have a maximum of 1, so
    `tail_dist * exp(maximum(log_ratios))` is the corresponding fit directly to the tail of
    `log_ratios`.

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
struct PSISResult{T,W<:AbstractArray{T},N,R,L,D}
    log_weights::W
    log_weights_norm::N
    reff::R
    tail_length::L
    tail_dist::D
end

function Base.propertynames(r::PSISResult)
    return [fieldnames(typeof(r))..., :weights, :nparams, :ndraws, :nchains, :pareto_shape]
end

function Base.getproperty(r::PSISResult, k::Symbol)
    if k === :weights
        log_weights = getfield(r, :log_weights)
        log_weights_norm = getfield(r, :log_weights_norm)
        return broadcast_last_dims(exp ∘ -, log_weights, log_weights_norm)
    elseif k === :nparams
        log_weights = getfield(r, :log_weights)
        return if ndims(log_weights) == 1
            1
        else
            prod(Base.Fix1(size, log_weights), param_dims(log_weights))
        end
    elseif k === :ndraws
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 1 ? length(log_weights) : size(log_weights, 1)
    elseif k === :nchains
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 3 ? size(log_weights, 2) : 1
    end
    k === :pareto_shape && return pareto_shape(r)
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
    ess = as_array(ess_is(r))
    npoints = r.nparams
    rows = map(SHAPE_DIAGNOSTIC_CATEGORIES) do (range, desc, cond)
        inds = findall(cond, k)
        count = length(inds)
        perc = 100 * count / npoints
        ess_min = if count == 0 || desc == "too few draws"
            missing
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
        "missing" => (),
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
        print(io, col_delim_tot, r.ess_min === missing ? "——" : floor(Int, r.ess_min))
    end
    return nothing
end

_pad_left(s, nchars) = " "^(nchars - length("$s")) * "$s"
_pad_right(s, nchars) = "$s" * " "^(nchars - length("$s"))

"""
    psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult
    psis!(log_ratios, reff = 1.0; kwargs...) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

While `psis` computes smoothed log weights out-of-place, `psis!` smooths them in-place.

# Arguments

  - `log_ratios`: an array of logarithms of importance ratios, with one of the following
    sizes:

      + `(draws,)`: a vector of draws for a single parameter from a single chain
      + `(draws, params)`: a matrix of draws for a multiple parameter from a single chain
      + `(draws, chains, params...)`: an array of draws for multiple parameters from
        multiple chains, e.g. as might be generated with Markov chain Monte Carlo.

  - `reff::Union{Real,AbstractArray}`: the ratio(s) of effective sample size of
    `log_ratios` and the actual sample size `reff = ess/(ndraws * nchains)`, used to account
    for autocorrelation, e.g. due to Markov chain Monte Carlo. If an array, it must have the
    size `(params...,)` to match `log_ratios`.

# Keywords

  - `warn=true`: If `true`, warning messages are delivered

# Returns

  - `result`: a [`PSISResult`](@ref) object containing the results of the Pareto-smoothing.

A warning is raised if the Pareto shape parameter ``k ≥ 0.7``. See [`PSISResult`](@ref) for
details and [`PSISPlots.paretoshapeplot`](@ref) for a diagnostic plot.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
psis, psis!

function psis(logr, reff=1; kwargs...)
    T = float(eltype(logr))
    logw = similar(logr, T)
    copyto!(logw, logr)
    return psis!(logw, reff; kwargs...)
end

function psis!(logw::AbstractVector, reff=1; warn::Bool=true)
    S = length(logw)
    reff_val = first(reff)
    M = tail_length(reff_val, S)
    if M < 5
        warn &&
            @warn "$M tail draws is insufficient to fit the generalized Pareto distribution. Total number of draws should in general exceed 25."
        return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, missing)
    end
    perm = partialsortperm(logw, (S - M):S)
    cutoff_ind = perm[1]
    tail_inds = @view perm[2:(M + 1)]
    logu = logw[cutoff_ind]
    logw_tail = @views logw[tail_inds]
    if !all(isfinite, logw_tail)
        warn &&
            @warn "Tail contains non-finite values. Generalized Pareto distribution cannot be reliably fit."
        return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, missing)
    end
    _, tail_dist = psis_tail!(logw_tail, logu, M)
    warn && check_pareto_shape(tail_dist)
    return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, tail_dist)
end
function psis!(logw::AbstractArray, reff=1; warn::Bool=true, kwargs...)
    T = typeof(float(one(eltype(logw))))
    # if an array defines custom indices (e.g. AbstractDimArray), we preserve them
    param_axes = map(Base.Fix1(axes, logw), param_dims(logw))

    # allocate containers
    reffs = similar(logw, eltype(reff), param_axes)
    reffs .= reff
    log_weights_norm = similar(logw, T, param_axes)
    tail_lengths = similar(logw, Int, param_axes)
    tail_dists = similar(logw, Union{Missing,GeneralizedPareto{T}}, param_axes)

    # call psis! in parallel for all parameters
    Threads.@threads for inds in CartesianIndices(param_axes)
        logw_i = vec(param_draws(logw, inds))
        result_i = psis!(logw_i, reffs[inds]; warn=false, kwargs...)
        log_weights_norm[inds] = result_i.log_weights_norm
        tail_lengths[inds] = result_i.tail_length
        tail_dists[inds] = result_i.tail_dist
    end

    # combine results
    result = PSISResult(logw, log_weights_norm, reffs, tail_lengths, tail_dists)

    # warn for bad shape
    warn && check_pareto_shape(result)
    return result
end

pareto_shape(::Missing) = missing
pareto_shape(dist::GeneralizedPareto) = dist.k
pareto_shape(r::PSISResult) = pareto_shape(getfield(r, :tail_dist))
pareto_shape(dists) = map(pareto_shape, dists)

check_pareto_shape(result::PSISResult) = check_pareto_shape(result.tail_dist)
function check_pareto_shape(dist::GeneralizedPareto)
    k = pareto_shape(dist)
    if k > 1
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 1. $VERY_BAD_SHAPE_SUMMARY"
    elseif k > 0.7
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 0.7. $BAD_SHAPE_SUMMARY"
    end
    return nothing
end
function check_pareto_shape(dists::AbstractArray{<:Union{Missing,GeneralizedPareto}})
    nmissing = count(ismissing, dists)
    ngt07 = count(x -> !(ismissing(x)) && pareto_shape(x) > 0.7, dists)
    ngt1 = iszero(ngt07) ? ngt07 : count(x -> !(ismissing(x)) && pareto_shape(x) > 1, dists)
    if ngt07 > ngt1
        @warn "$(ngt07 - ngt1) parameters had Pareto shape values 0.7 < k ≤ 1. $BAD_SHAPE_SUMMARY"
    end
    if ngt1 > 0
        @warn "$ngt1 parameters had Pareto shape values k > 1. $VERY_BAD_SHAPE_SUMMARY"
    end
    if nmissing > 0
        @warn "For $nmissing parameters, the generalized Pareto distribution could not be fit to the tail draws. Total number of draws should in general exceed 25, and the tail draws must be finite."
    end
    return nothing
end

tail_length(reff, S) = min(cld(S, 5), ceil(Int, 3 * sqrt(S / reff)))

function psis_tail!(logw, logμ, M=length(logw))
    T = eltype(logw)
    logw_max = logw[M]
    # to improve numerical stability, we first shift the log-weights to have a maximum of 0,
    # equivalent to scaling the weights to have a maximum of 1.
    μ_scaled = exp(logμ - logw_max)
    w = (logw .= exp.(logw .- logw_max))
    tail_dist_scaled = fit_gpd(w; sorted=true, μ=μ_scaled)
    tail_dist_adjusted = prior_adjust_shape(tail_dist_scaled, M)
    # undo the scaling
    k = tail_dist_adjusted.k
    if isfinite(k)
        p = uniform_probabilities(T, M)
        @inbounds for i in eachindex(logw, p)
            # undo scaling in the log-weights
            logw[i] = min(log(quantile(tail_dist_adjusted, p[i])), 0) + logw_max
        end
    end
    return logw, tail_dist_adjusted
end
