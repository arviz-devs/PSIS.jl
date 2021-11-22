module PSIS

using Distributions: Distributions
using LinearAlgebra: dot
using LogExpFunctions: logsumexp, softmax!
using Printf: @sprintf
using Statistics: mean, median, quantile
using StatsBase: StatsBase

export psis, psis!

include("utils.jl")
include("generalized_pareto.jl")

"""
    psis(log_ratios, r_eff; kwargs...) -> (log_weights, k)

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

See [`psis!`](@ref) for a version that smoothes the ratios in-place.

# Arguments

  - `log_ratios`: an array of logarithms of importance ratios, with one of the following
    sizes:
    
      + `(ndraws,)`: a vector of draws for a single parameter from a single chain
      + `(nparams, ndraws)`: a matrix of draws for a multiple parameter from a single chain
      + `(nparams, ndraws, nchains)`: an array of draws for multiple parameters from
        multiple chains, e.g. as might be generated with Markov chain Monte Carlo.

  - `r_eff`: the ratio of effective sample size of `log_ratios` and the actual sample size,
    used to account for autocorrelation, e.g. due to Markov chain Monte Carlo. If the ratios
    are known to be uncorrelated, then provide `r_eff=ones(nparams)`.

# Keywords

  - `sorted=issorted(vec(log_ratios))`: whether `log_ratios` are already sorted. Only
    accepted if `nparams==1`.
  - `normalize=false`: whether to normalize the log weights so that the resulting weights
    for a given parameter sum to one.
  - `improved=false`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009], which is also used in
    [^VehtariSimpson2021].

# Returns

  - `log_weights`: an array of smoothed log weights of the same size as `log_ratios`
  - `k`: for each parameter, the estimated shape parameter ``k`` of the generalized Pareto
    distribution, which is useful for diagnosing the distribution of importance ratios.
    See details below.

# Diagnostic

The shape parameter ``k`` of the generalized Pareto distribution can be used to diagnose
reliability and convergence of estimates using the importance weights [^VehtariSimpson2021]:

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

A warning is raised if ``k ≥ 0.7``.

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
function psis(logr, r_eff; kwargs...)
    T = float(eltype(logr))
    logw = copyto!(similar(logr, T), logr)
    return psis!(logw, r_eff; kwargs...)
end

"""
    psis!(args, r_eff; kwargs...)

In-place compute Pareto smoothed importance sampling (PSIS) log weights.

See [`psis`](@ref) for an out-of-place version and for description of arguments and return
values.
"""
function psis!(
    logw::AbstractVector, r_eff; sorted=issorted(logw), normalize=false, improved=false
)
    T = eltype(logw)
    S = length(logw)
    k_hat = T(Inf)

    M = tail_length(only(r_eff), S)
    if M < 5
        @warn "Insufficient tail draws to fit the generalized Pareto distribution."
    else
        perm = sorted ? eachindex(logw) : sortperm(logw)
        @inbounds logw_max = logw[last(perm)]
        icut = S - M
        tail_range = (icut + 1):S

        @inbounds logw_tail = @views logw[perm[tail_range]]
        if logw_max - first(logw_tail) < eps(eltype(logw_tail)) / 100
            @warn "Cannot fit the generalized Pareto distribution because all tail " *
                "values are the same"
        else
            logw_tail .-= logw_max
            @inbounds logu = logw[perm[icut]] - logw_max

            _, k_hat = psis_tail!(logw_tail, logu, M, improved)
            logw_tail .+= logw_max

            check_pareto_k(k_hat)
        end
    end

    if normalize
        logw .-= logsumexp(logw)
    end

    return logw, k_hat
end
function psis!(logw::AbstractArray, r_eff; kwargs...)
    # support both 2D and 3D arrays, flattening the final dimension
    _, k_hat = psis!(vec(selectdim(logw, 1, 1)), r_eff[1]; kwargs...)
    # for arrays with named dimensions, this pattern ensures k_hat has the same names
    k_hats = similar(view(logw, :, ntuple(_ -> 1, ndims(logw) - 1)...), eltype(k_hat))
    k_hats[1] = k_hat
    Threads.@threads for i in eachindex(k_hats, r_eff)
        _, k_hats[i] = psis!(vec(selectdim(logw, 1, i)), r_eff[i]; kwargs...)
    end
    return logw, k_hats
end

function check_pareto_k(k)
    if k ≥ 1
        @warn "Pareto k=$(@sprintf("%.2g", k)) ≥ 1. Resulting importance sampling " *
            "estimates are likely to be unstable and are unlikely to converge with " *
            "additional samples."
    elseif k ≥ 0.7
        @warn "Pareto k=$(@sprintf("%.2g", k)) ≥ 0.7. Resulting importance sampling " *
            "estimates are likely to be unstable."
    end
    return nothing
end

tail_length(r_eff, S) = min(cld(S, 5), ceil(Int, 3 * sqrt(S / r_eff)))

function psis_tail!(logw, logμ, M=length(logw), improved=false)
    T = eltype(logw)
    logw_max = logw[M]
    # to improve numerical stability, we first scale the log-weights to have a maximum of 1,
    # equivalent to shifting the log-weights to have a maximum of 0.
    μ_scaled = exp(logμ - logw_max)
    w = (logw .= exp.(logw .- logw_max))
    tail_dist_scaled = StatsBase.fit(
        GeneralizedParetoKnownMu(μ_scaled),
        w;
        sorted=true,
        improved=improved,
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

end
