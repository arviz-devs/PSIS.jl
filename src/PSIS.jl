module PSIS

using Distributions: Distributions
using LinearAlgebra: dot
using LogExpFunctions: logsumexp
using Printf: @sprintf
using Statistics: mean, median, quantile
using StatsBase: StatsBase

export psis, psis!

include("utils.jl")
include("generalized_pareto.jl")

"""
    psis(log_ratios, r_eff; kwargs...) -> (log_weights, k)

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

See [`psis!`](@ref) for version that smoothes the ratios in-place.

# Arguments

  - `log_ratios`: an array of logarithms of importance ratios.
  - `r_eff`: the ratio of effective sample size of `log_ratios` and the actual sample size,
    used to correct for autocorrelation due to MCMC. `r_eff=1` should be used if the ratios
    were sampled independently.

# Keywords

  - `sorted=issorted(log_ratios)`: whether `log_ratios` are already sorted.
  - `normalize=false`: whether to normalize the log weights so that
    `sum(exp.(low_weights)) ≈ 1`.
  - `improved=false`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009], which is also used in
    [^VehtariSimpson2021].

# Returns

  - `log_weights`: an array of smoothed log weights
  - `k`: the estimated shape parameter ``k`` of the generalized Pareto distribution, which
    is useful for diagnosing the distribution of importance ratios. See details below.

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
function psis(logr, r_eff=1.0; kwargs...)
    T = float(eltype(logr))
    logw = copyto!(similar(logr, T), logr)
    return psis!(logw, r_eff; kwargs...)
end

"""
    psis!(args...; kwargs...)

In-place compute Pareto smoothed importance sampling (PSIS) log weights.

See [`psis`](@ref) for an out-of-place version and for description of arguments and return
values.
"""
function psis!(logw, r_eff=1.0; sorted=issorted(logw), normalize=false, improved=false)
    T = eltype(logw)
    S = length(logw)
    k_hat = T(Inf)

    M = tail_length(r_eff, S)
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

function psis_tail!(logw, logu, M=length(logw), improved=false)
    T = eltype(logw)
    u = exp(logu)
    w = (logw .= exp.(logw))
    d_hat = StatsBase.fit(GeneralizedParetoKnownMu(u), w; sorted=true, improved=improved)
    d_hat = prior_adjust_shape(d_hat, M)
    k_hat = Distributions.shape(d_hat)
    if isfinite(k_hat)
        p = uniform_probabilities(T, M)
        logw .= min.(log.(quantile.(Ref(d_hat), p)), zero(T))
    end
    return logw, k_hat
end

end
