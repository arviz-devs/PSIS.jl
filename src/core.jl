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
  - if ``k < \\frac{1}{2}``, then the importance ratio distributon has finite variance, and
    the central limit theorem holds. As ``k`` approaches the upper bound, IS becomes less
    reliable, while PSIS still works well but with a higher RMSE.
  - if ``\\frac{1}{2} ≤ k < 0.7``, then the variance is infinite, and IS can behave quite
    poorly. However, PSIS works well in this regime.
  - if ``0.7 ≤ k < 1``, then it quickly becomes impractical to collect enough importance
    weights to reliably compute estimates, and importance sampling is not recommended.
  - if ``k ≥ 1``, then neither the variance nor the mean of the raw importance ratios
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
    return getfield(r, k)
end

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult)
    println(io, typeof(r), ":")
    print(io, "    pareto_shape: ", r.pareto_shape)
    return nothing
end

"""
    psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

See [`psis!`](@ref) for a version that smoothes the ratios in-place.

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

  - `sorted=issorted(vec(log_ratios))`: whether `log_ratios` are already sorted. Only
    accepted if `nparams==1`.
  - `improved=false`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009], which is also used in
    [^VehtariSimpson2021].

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
function psis(logr, reff=1; kwargs...)
    T = float(eltype(logr))
    logw = similar(logr, T)
    copyto!(logw, logr)
    return psis!(logw, reff; kwargs...)
end

"""
    psis!(args, reff = 1.0; kwargs...)

In-place compute Pareto smoothed importance sampling (PSIS) log weights.

See [`psis`](@ref) for an out-of-place version and for description of arguments and return
values.
"""
function psis!(logw::AbstractVector, reff=1; sorted=issorted(logw), improved=false)
    S = length(logw)
    reff_val = first(reff)
    M = tail_length(reff_val, S)
    if M < 5
        @warn "Insufficient tail draws to fit the generalized Pareto distribution."
        return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, missing)
    end
    perm = sorted ? collect(eachindex(logw)) : sortperm(logw)
    icut = S - M
    tail_range = (icut + 1):S
    @inbounds logw_tail = @views logw[perm[tail_range]]
    @inbounds logu = logw[perm[icut]]
    _, tail_dist = psis_tail!(logw_tail, logu, M, improved)
    check_pareto_shape(tail_dist)
    return PSISResult(logw, LogExpFunctions.logsumexp(logw), reff_val, M, tail_dist)
end
function psis!(logw::AbstractArray, reff=1; kwargs...)
    Tdist = Union{Distributions.GeneralizedPareto{eltype(logw)},Missing}
    logw_firstcol = view(logw, :, ntuple(_ -> 1, ndims(logw) - 1)...)
    reff_vec = reff isa Number ? fill!(similar(logw_firstcol), reff) : reff
    # support both 2D and 3D arrays, flattening the final dimension
    r1 = psis!(vec(selectdim(logw, 1, 1)), reff_vec[1]; kwargs...)
    # for arrays with named dimensions, this pattern ensures tail_lengths and tail_dists
    # have the same names
    logw_norm = similar(logw_firstcol)
    logw_norm[1] = r1.log_weights_norm
    tail_lengths = similar(logw_firstcol, Int)
    tail_lengths[1] = r1.tail_length
    tail_dists = similar(logw_firstcol, Tdist)
    tail_dists[1] = r1.tail_dist
    Threads.@threads for i in eachindex(tail_dists, reff_vec, tail_lengths, tail_dists)
        ri = psis!(vec(selectdim(logw, 1, i)), reff_vec[i]; warn=false, kwargs...)
        logw_norm[i] = ri.log_weights_norm
        tail_lengths[i] = ri.tail_length
        tail_dists[i] = ri.tail_dist
    end
    result = PSISResult(logw, logw_norm, reff_vec, tail_lengths, map(identity, tail_dists))
    return result
end

pareto_shape(::Missing) = missing
pareto_shape(dist::Distributions.GeneralizedPareto) = Distributions.shape(dist)
pareto_shape(r::PSISResult) = pareto_shape(getfield(r, :tail_dist))
pareto_shape(dists) = map(pareto_shape, dists)

function check_pareto_shape(dist::Distributions.GeneralizedPareto)
    ξ = pareto_shape(dist)
    if ξ ≥ 1
        @warn "Pareto shape=$(@sprintf("%.2g", ξ)) ≥ 1. Resulting importance sampling " *
            "estimates are likely to be unstable and are unlikely to converge with " *
            "additional samples."
    elseif ξ ≥ 0.7
        @warn "Pareto shape=$(@sprintf("%.2g", ξ)) ≥ 0.7. Resulting importance sampling " *
            "estimates are likely to be unstable."
    end
    return nothing
end

tail_length(reff, S) = min(cld(S, 5), ceil(Int, 3 * sqrt(S / reff)))

function psis_tail!(logw, logμ, M=length(logw), improved=false)
    T = eltype(logw)
    logw_max = logw[M]
    # to improve numerical stability, we first scale the log-weights to have a maximum of 1,
    # equivalent to shifting the log-weights to have a maximum of 0.
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
