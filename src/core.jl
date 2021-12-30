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
const MISSING_SHAPE_SUMMARY = "Total number of draws should in general exceed 25."

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
    `log_weights`

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

See [`paretoshapeplot`](@ref) for a diagnostic plot.

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
        return exp.(getfield(r, :log_weights) .- getfield(r, :log_weights_norm))
    elseif k === :nparams
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 1 ? 1 : size(log_weights, 1)
    elseif k === :ndraws
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 1 ? length(log_weights) : size(log_weights, 2)
    elseif k === :nchains
        log_weights = getfield(r, :log_weights)
        return size(log_weights, 3)
    end
    k === :pareto_shape && return pareto_shape(r)
    k === :ess && return ess_is(r)
    return getfield(r, k)
end

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    npoints = r.nparams
    nchains = r.nchains
    println(
        io, "PSISResult with $npoints parameters, $(r.ndraws) draws, and $nchains chains"
    )
    return _print_pareto_shape_summary(io, r; newline_at_end=false)
end

function pareto_shape_summary(r::PSISResult; kwargs...)
    return _print_pareto_shape_summary(stdout, r; kwargs...)
end
function _print_pareto_shape_summary(io::IO, r::PSISResult; kwargs...)
    ξ = as_array(pareto_shape(r))
    ess = as_array(ess_is(r))
    npoints = r.nparams
    rows = map(SHAPE_DIAGNOSTIC_CATEGORIES) do (range, desc, cond)
        inds = findall(cond, ξ)
        count = length(inds)
        perc = 100 * count / npoints
        ess_min = if count == 0 || desc == "too few draws"
            missing
        else
            minimum(view(ess, inds))
        end
        return (range=range, desc=desc, count_perc=(count, perc), ess_min=ess_min)
    end

    return PrettyTables.pretty_table(
        io,
        collect(rows);
        title="Pareto shape (k) diagnostic values:",
        title_crayon=PrettyTables.crayon"",
        header=["", "", "Count", "Min. ESS"],
        alignment=[:r, :l, :l, :l],
        alignment_anchor_regex=Dict(3 => [r" \("]),
        filters_row=((data, i) -> data[i].count_perc[1] > 0,),
        formatters=(
            (data, i, j) -> j == 3 ? "$(data[1]) ($(round(data[2]; digits=1))%)" : data,
            (data, i, j) -> j == 4 ? (ismissing(data) ? "——" : floor(Int, data)) : data,
        ),
        highlighters=(
            PrettyTables.hl_cell([(2, 3)], PrettyTables.crayon"yellow"),
            PrettyTables.hl_cell([(3, 3)], PrettyTables.crayon"light_red bold"),
            PrettyTables.hl_cell([(4, 3)], PrettyTables.crayon"red bold"),
        ),
        crop=:horizontal,
        hlines=:none,
        vlines=:none,
        kwargs...,
    )
end

function _promote_result_type(::Type{PSISResult{T,W,N,R,L,D}}) where {T,W,N,R,L,D}
    return PSISResult{
        T,W,N,R,L,D2
    } where {D2<:Union{D,Missing,Distributions.GeneralizedPareto{T}}}
end

"""
    psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult
    psis!(log_ratios, reff = 1.0; kwargs...) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

While `psis` computes smoothed log weights out-of-place, `psis!` smooths them in-place.

# Arguments

  - `log_ratios`: an array of logarithms of importance ratios, with one of the following
    sizes:
    
      + `(ndraws,)`: a vector of draws for a single parameter from a single chain
      + `(nparams, ndraws)`: a matrix of draws for a multiple parameter from a single chain
      + `(nparams, ndraws, nchains)`: an array of draws for multiple parameters from
        multiple chains, e.g. as might be generated with Markov chain Monte Carlo.

  - `reff::Union{Real,AbstractVector}`: the ratio(s) of effective sample size of
    `log_ratios` and the actual sample size `reff = ess/(ndraws * nchains)`, used to account
    for autocorrelation, e.g. due to Markov chain Monte Carlo.

# Keywords

  - `improved=false`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009], which is also used in
    [^VehtariSimpson2021].
  - `warn=true`: If `true`, warning messages are delivered

# Returns

  - `result`: a [`PSISResult`](@ref) object containing the results of the Pareto-smoothing.

A warning is raised if the Pareto shape parameter ``k ≥ 0.7``. See [`PSISResult`](@ref) for
details and [`paretoshapeplot`](@ref) for a diagnostic plot.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
[^ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009)
    A New and Efficient Estimation Method for the Generalized Pareto Distribution,
    Technometrics, 51:3, 316-325,
    DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
[^Zhang2010]: Jin Zhang (2010) Improving on Estimation for the Generalized Pareto Distribution,
    Technometrics, 52:3, 335-339,
    DOI: [10.1198/TECH.2010.09206](https://doi.org/10.1198/TECH.2010.09206)
"""
psis, psis!

function psis(logr, reff=1; kwargs...)
    T = float(eltype(logr))
    logw = similar(logr, T)
    copyto!(logw, logr)
    return psis!(logw, reff; kwargs...)
end

function psis!(
    logw::AbstractVector,
    reff=1;
    sorted::Bool=false, # deprecated
    improved::Bool=false,
    warn::Bool=true,
)
    S = length(logw)
    reff_val = first(reff)
    M = tail_length(reff_val, S)
    if M < 5
        warn &&
            @warn "$M tail draws is insufficient to fit the generalized Pareto distribution. $MISSING_SHAPE_SUMMARY"
        return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, missing)
    end
    perm = partialsortperm(logw, (S - M):S)
    cutoff_ind = perm[1]
    tail_inds = @view perm[2:M + 1]
    logu = logw[cutoff_ind]
    logw_tail = @views logw[tail_inds]
    _, tail_dist = psis_tail!(logw_tail, logu, M, improved)
    warn && check_pareto_shape(tail_dist)
    return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, tail_dist)
end
function psis!(logw::AbstractArray, reff=1; warn::Bool=true, kwargs...)
    # allocate containers, calling psis! for first parameter to determine eltype
    logw_firstdraw = first_draw(logw)
    reffs = reff isa Number ? fill!(similar(logw_firstdraw), reff) : reff
    r1 = psis!(vec(param_draws(logw, 1)), reffs[1]; warn=false, kwargs...)
    results = similar(logw_firstdraw, _promote_result_type(typeof(r1)))
    i, inds = Iterators.peel(eachindex(results, reffs))
    results[i] = r1
    # call psis! for remaining parameters
    Threads.@threads for i in collect(inds)
        results[i] = psis!(vec(param_draws(logw, i)), reffs[i]; warn=false, kwargs...)
    end
    # combine results
    logw_norms = map(r -> r.log_weights_norm, results)
    tail_lengths = map(r -> r.tail_length, results)
    tail_dists = map(r -> r.tail_dist, results)
    result = PSISResult(logw, logw_norms, reffs, tail_lengths, tail_dists)
    # warn for bad shape
    warn && check_pareto_shape(result)
    return result
end

pareto_shape(::Missing) = missing
pareto_shape(dist::Distributions.GeneralizedPareto) = Distributions.shape(dist)
pareto_shape(r::PSISResult) = pareto_shape(getfield(r, :tail_dist))
pareto_shape(dists) = map(pareto_shape, dists)

check_pareto_shape(result::PSISResult) = check_pareto_shape(result.tail_dist)
function check_pareto_shape(dist::Distributions.GeneralizedPareto)
    k = pareto_shape(dist)
    if k > 1
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 1. $VERY_BAD_SHAPE_SUMMARY"
    elseif k > 0.7
        @warn "Pareto shape k = $(@sprintf("%.2g", k)) > 0.7. $BAD_SHAPE_SUMMARY"
    end
    return nothing
end
function check_pareto_shape(
    dists::AbstractVector{<:Union{Missing,Distributions.GeneralizedPareto}}
)
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
        @warn "$nmissing parameters had insufficient tail draws to fit the generalized Pareto distribution. $MISSING_SHAPE_SUMMARY"
    end
    return nothing
end

tail_length(reff, S) = min(cld(S, 5), ceil(Int, 3 * sqrt(S / reff)))

function psis_tail!(logw, logμ, M=length(logw), improved=false)
    T = eltype(logw)
    logw_max = logw[M]
    # to improve numerical stability, we first shift the log-weights to have a maximum of 0,
    # equivalent to scaling the weights to have a maximum of 1.
    μ_scaled = exp(logμ - logw_max)
    w = (logw .= exp.(logw .- logw_max))
    tail_dist_scaled = StatsBase.fit(
        GeneralizedParetoKnownMu(μ_scaled), w; sorted=true, improved=improved
    )
    tail_dist_adjusted = prior_adjust_shape(tail_dist_scaled, M)
    # undo the scaling
    ξ = Distributions.shape(tail_dist_adjusted)
    if isfinite(ξ)
        p = uniform_probabilities(T, M)
        @inbounds for i in eachindex(logw, p)
            # undo scaling in the log-weights
            logw[i] = min(log(_quantile(tail_dist_adjusted, p[i])), 0) + logw_max
        end
    end
    # undo scaling for the tail distribution
    tail_dist = scale(tail_dist_adjusted, exp(logw_max))
    return logw, tail_dist
end
