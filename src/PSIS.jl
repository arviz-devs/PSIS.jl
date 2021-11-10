module PSIS

using Distributions: Distributions
using LogExpFunctions: logsumexp
using Statistics: mean, quantile
using StatsBase: StatsBase
using LinearAlgebra: dot
using Printf: @sprintf

export PSISResult, psis, psis!

include("utils.jl")
include("generalized_pareto.jl")

"""
    PSISResult

Result of Pareto-smoothed importance sampling (PSIS).

# Properties

  - `log_weights`: unnormalized Pareto-smoothed log weights
  - `weights`: normalized Pareto-smoothed weights (allocates a copy)
  - `ndraws`: length of `log_weights` and `weights`
  - `pareto_k`: Pareto ``k=ξ`` shape parameter
  - `r_eff`: the ratio of the effective sample size of the unsmoothed importance ratios and
    the actual sample size.
  - `tail_length`: length of the upper tail of `log_weights` that was smoothed
  - `tail_dist`: the generalized Pareto distribution that was fit to the tail of `log_weights`

See [`psis`](@ref) for a description of how to use `pareto_k` as a diagnostic.
"""
struct PSISResult{
    T,W<:AbstractArray{T},R<:Real,D<:Union{Missing,Distributions.GeneralizedPareto{T}}
}
    log_weights::W
    r_eff::R
    tail_length::Int
    tail_dist::D
end

function Base.getproperty(r::PSISResult, k::Symbol)
    if k === :weights
        log_weights = getfield(r, :log_weights)
        return exp.(log_weights .- logsumexp(log_weights))
    end
    k === :ndraws && return length(getfield(r, :log_weights))
    k === :pareto_k && return pareto_k(getfield(r, :tail_dist))
    return getfield(r, k)
end

Base.propertynames(r::PSISResult) = [fieldnames(typeof(r))..., :weights, :pareto_k, :ndraws]

function Base.show(io::IO, ::MIME"text/plain", r::PSISResult{T}) where {T}
    println(io, typeof(r), ":")
    println(io, "    ndraws: ", r.ndraws)
    println(io, "    r_eff: ", r.r_eff)
    print(io, "    pareto_k: ", r.pareto_k)
    return nothing
end

"""
    psis(log_ratios, r_eff; sorted=issorted(log_ratios)) -> PSISResult

Compute Pareto smoothed importance sampling (PSIS) log weights [^VehtariSimpson2021].

See [`psis!`](@ref) for version that smoothes the ratios in-place.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
# Arguments

  - `log_ratios`: an array of logarithms of importance ratios.
  - `r_eff`: the ratio of effective sample size of `log_ratios` and the actual sample size,
    used to correct for autocorrelation due to MCMC. `r_eff=1` should be used if the ratios
    were sampled independently.

# Returns

  - `result::PSISResult`: The result of PSIS. See [`PSISResult`](@ref) for a description of
    the properties.

# Diagnostic

The shape parameter ``k`` of the generalized Pareto distribution (accessed by
`result.pareto_k`) can be used to diagnose reliability and convergence of estimates using
the importance weights [^VehtariSimpson2021]:

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
"""
function psis(logr, r_eff; kwargs...)
    T = float(eltype(logr))
    logw = copyto!(similar(logr, T), logr)
    return psis!(logw, r_eff; kwargs...)
end

"""
    psis!(log_ratios, r_eff; sorted=issorted(log_ratios)) -> PSISResult

In-place compute Pareto smoothed importance sampling (PSIS) log weights.

The log importance ratios are overwritten by the (unnormalized) log importance weights,
which are also contained within the returned [`PSISResult`](@ref) object.

See [`psis`](@ref) for an out-of-place version and for description of arguments and return
values.
"""
function psis!(logw, r_eff; sorted::Bool=issorted(logw))
    S = length(logw)
    M = tail_length(r_eff, S)

    if M < 5
        @warn "Insufficient tail draws to fit the generalized Pareto distribution."
        return PSISResult(logw, r_eff, M, missing)
    end

    @inbounds logw_max = logw[last(perm)]
    perm = sorted ? collect(eachindex(logw_vec)) : sortperm(logw_vec)
    icut = S - M
    tail_range = (icut + 1):S
    @inbounds logw_tail = @views logw[perm[tail_range]]
    @inbounds logμ = logw[perm[icut]]
    _, tail_dist = psis_tail!(logw_tail, logμ; sorted=true)
    check_tail_dist(tail_dist)
    return PSISResult(logw, r_eff, M, tail_dist)
end

pareto_k(d::Distributions.GeneralizedPareto) = d.ξ
pareto_k(::Missing) = missing

function check_tail_dist(d::Distributions.GeneralizedPareto)
    k = pareto_k(d)
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

function psis_tail!(logw, logμ; sorted=issorted(logw))
    T = eltype(logw)
    M = length(logw)
    sorted || sort!(logw)
    logw_max = @inbounds logw[M]
    # shift to fit zero-centered GPD and scale for increased numerical stability
    μ_scaled = exp(logμ - logw_max)
    # reuse storage
    w = (logw .= exp.(logw .- logw_max) .- μ_scaled)
    # fit tail distribution
    method = EmpiricalBayesEstimate()
    tail_dist = StatsBase.fit!(GeneralizedParetoKnownMu(zero(T)), w, method; sorted=true)
    if isfinite(pareto_k(tail_dist))
        p = uniform_probabilities(T, M)
        # compute smoothed log-weights, undoing shift-and-scale
        logw .= min.(log.(quantile.(Ref(tail_dist), p) .+ μ_scaled), 0) .+ logw_max
    end
    # undo shift-and-scale
    tail_dist_trans = shift_then_scale(tail_dist, μ_scaled, exp(logw_max))
    return logw, tail_dist_trans
end

end
