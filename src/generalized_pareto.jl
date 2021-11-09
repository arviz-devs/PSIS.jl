# compute (d + shift) * scale but return GeneralizedPareto instead of LocationScale
function shift_then_scale(d::Distributions.GeneralizedPareto, shift, scale)
    return Distributions.GeneralizedPareto((d.μ + shift) * scale, d.σ * scale, d.ξ)
end

struct EmpiricalBayesEstimate end

struct GeneralizedParetoKnownMu{T} <: Distributions.IncompleteDistribution
    μ::T
end

"""
    StatsBase.fit!(
        d::GeneralizedParetoKnownMu,
        x,
        method::EmpiricalBayesEstimate;
        kwargs...,
    ) -> Distributions.GeneralizedPareto

Estimate a generalized Pareto distribution (GPD) given the points `x` and mean `d.μ`.

Compute an empirical Bayes estimate of the scale parameter ``σ`` and shape parameter ``ξ``
of the GPD given the data ``x`` and known mean ``μ`` using the method described in
[^ZhangStephens2009]. The method assumes all elements of ``x`` are greater than ``μ``.
If necessary, `x` is modified in-place during the fitting procedure.

Note that ``ξ`` here is related to ``k`` in [^ZhangStephens2009] by ``ξ = -k``.

# Keywords

  - `sorted=false`: whether `x` is already sorted. If `false`, it will be sorted in-place.
  - `min_points=30`: default number of points to use to compute estimate.
    Instead of `min_points=20` as recommended by [^ZhangStephens2009], we use 30 points
    as in the loo package. [^ZhangStephens2009] notes that the estimator is not sensitive
    to this choice.
  - `adjust_prior=true`: whether to apply the weakly informative Gaussian prior on ``ξ``
    suggested by [^VehtariSimpson2021] to reduce variance in the estimate of ``ξ``.

[^ZhangStephens2009]: Zhang J & Stephens M A. (2009).
    A new and efficient estimation method for the generalized Pareto distribution,
    Technometrics, 51:3, 316-325,
    DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
function StatsBase.fit!(
    d::GeneralizedParetoKnownMu,
    x,
    ::EmpiricalBayesEstimate;
    sorted=false,
    min_points=30,
    adjust_prior=true,
)
    n = length(x)
    μ = d.μ
    T = Base.promote_eltype(x, μ)
    sorted || sort!(x)
    # shift the data to fit the zero-centered GPD
    if !iszero(μ)
        x .-= μ
    end
    if last(x) ≤ first(x) * exp(eps(T) / 100)
        # support is nearly a point. solution is not unique; any solution satisfying the
        # constraints σ/ξ ≈ 0 and ξ < 0 is acceptable
        return Distributions.GeneralizedPareto(T(μ), one(T), -T(Inf))
    end
    # fit the zero-centered GPD
    m = min_points + floor(Int, sqrt(n))
    θ_hat = _fit_θ(x, m) # θ = σ / ξ
    ξ_hat = _fit_ξ(x, θ_hat)
    σ_hat = _fit_σ(θ_hat, ξ_hat)
    # NOTE: the paper is ambiguous whether the adjustment is applied to ξ_hat
    # before or after computing σ_hat. From private discussion with Aki Vehtari,
    # adjusting afterwards produces better results.
    ξ_hat = adjust_prior ? prior_adjust_ξ(ξ_hat, n) : ξ_hat
    return Distributions.GeneralizedPareto(μ, σ_hat, ξ_hat)
end

# estimate θ̂ = ∫θp(θ|x)dθ using quadrature over m grid points
# uniformly sampled over the empirical prior
function _fit_θ(x, m)
    T = float(eltype(x))
    n = length(x)

    # construct empirical prior on y = 1/x[n] - θ
    x_star = quartile(x, 1)
    σ_star = inv(6 * x_star)
    ξ_star = 1//2
    y_prior = Distributions.GeneralizedPareto(zero(T), σ_star, ξ_star)

    # quadrature points uniformly spaced on the quantiles of the θ prior
    p = uniform_probabilities(T, m)
    @inbounds θ = inv(x[n]) .- quantile.(Ref(y_prior), p)

    # compute mean θ over the m quadrature points
    # with weights as the normalized profile likelihood 
    lθ = profile_loglikelihood.(θ, Ref(x), n)
    lθ_norm = logsumexp(lθ)
    θ_hat = @inbounds sum(1:m) do j
        wⱼ = exp(lθ[j] - lθ_norm)
        return θ[j] * wⱼ
    end

    return θ_hat
end

# Zhang & Stephens, Eq 7
function _fit_ξ(x, θ_hat)
    nθ_hat = -θ_hat
    return mean(xᵢ -> log1p(nθ_hat * xᵢ), x)
end

_fit_σ(θ_hat, ξ_hat) = -ξ_hat / θ_hat

# compute likelihood p(x|θ,ξ), estimating ξ from θ
function profile_loglikelihood(θ, x, n=length(x))
    # estimate ξ given θ
    nξ_est = -_fit_ξ(x, θ)
    return n * (log(θ / nξ_est) + nξ_est - 1)
end

# reduce variance of estimated ξ using a weakly informative Gaussian prior
# centered at `ξ_prior` corresponding to `nobs` observations
# Vehtari et al, Appendix C
prior_adjust_ξ(ξ, n, ξ_prior=1//2, nobs=10) = (n * ξ + nobs * ξ_prior) / (n + nobs)
