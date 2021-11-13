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
struct PSISResult{T,W<:AbstractArray{T},R,L,D} <: AbstractISResult
    log_weights::W
    r_eff::R
    tail_length::L
    tail_dist::D
end

function Base.propertynames(r::PSISResult)
    return [fieldnames(typeof(r))..., :weights, :nparams, :ndraws, :nchains, :pareto_k]
end

function Base.show(io::IO, mime::MIME"text/plain", r::PSISResult)
    invoke(Base.show, Tuple{IO,typeof(mime),AbstractISResult}, io, mime, r)
    print(io, "\n    pareto_k: ", r.pareto_k)
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
    M = tail_length(r_eff[1], S)

    if M < 5
        @warn "Insufficient tail draws to fit the generalized Pareto distribution."
        return PSISResult(logw, r_eff, M, missing)
    end

    logw_vec = vec(logw)
    perm = sorted ? collect(eachindex(logw_vec)) : sortperm(logw_vec)
    @inbounds logw_max = logw_vec[last(perm)]
    icut = S - M
    tail_range = (icut + 1):S
    @inbounds logw_tail = @views logw_vec[perm[tail_range]]
    @inbounds logμ = logw_vec[perm[icut]]
    _, tail_dist = psis_tail!(logw_tail, logμ; sorted=true)
    check_tail_dist(tail_dist)

    return PSISResult(logw, r_eff, M, tail_dist)
end
function psis!(logw::AbstractArray{T,3}, r_eff::AbstractVector) where {T}
    nparams = size(logw, 1)
    @assert nparams == length(r_eff)
    tail_dists = Vector{Distributions.GeneralizedPareto{T}}(undef, nparams)
    tail_lengths = Vector{Int}(undef, nparams)
    Threads.@threads for i in 1:nparams
        logwᵢ = @views logw[i, :, :]
        res = psis!(logwᵢ, r_eff[i])
        tail_dists[i] = res.tail_dist
        tail_lengths[i] = res.tail_length
    end
    return PSISResult(logw, r_eff, tail_lengths, tail_dists)
end

pareto_k(d::Distributions.GeneralizedPareto) = d.ξ
pareto_k(::Missing) = missing
pareto_k(ds::AbstractVector) = map(pareto_k, ds)
function pareto_k(res::PSISResult)
    kvals = pareto_k(getfield(res, :tail_dist))
    kvals isa AbstractVector || return kvals
    # for arrays with named axes and dimensions, this preserves the names
    # in the result
    log_weights = getfield(res, :log_weights)
    k = similar(view(log_weights, :, 1, 1))
    copyto!(k, kvals)
    return k
end

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
