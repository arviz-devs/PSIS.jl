# Note: These internal functions are here to avoid a dependency on Distributions.jl,
# which currently does not implement GPD fitting anyways. They are not methods of
# functions in Statistics/StatsBase

"""
    GeneralizedPareto{T<:Real}

The (zero-centered) generalized Pareto distribution.

# Constructor

    GeneralizedPareto(σ, k)

Construct the generalized Pareto distribution (GPD) with scale parameter ``σ`` and shape
parameter ``k``. Note that this ``k`` is equal to the commonly used shape parameter ``ξ``.
This is the same parameterization used by [^VehtariSimpson2021] and is related to that used
by [^ZhangStephens2009] as ``k \\mapsto -k``.
"""
struct GeneralizedPareto{T}
    σ::T
    k::T
end
GeneralizedPareto(σ, k) = GeneralizedPareto(Base.promote(σ, k)...)

"""
    quantile(d::GeneralizedPareto, p)

Compute the ``p``-quantile of the generalized Pareto distribution `d`.
"""
@inline function quantile(d::GeneralizedPareto, p)
    k = d.k
    z = -log1p(-p)
    return iszero(k) ? d.σ * z : expm1(k * z) * (d.σ / k)
end

"""
    fit(T::Type{<:GeneralizedPareto}, x; kwargs...) -> ::T

Estimate a generalized Pareto distribution (GPD) given the points `x`.

Compute an empirical Bayes estimate of the parameters of the (zero-centered) GPD given the
data `x` using the method described in [^ZhangStephens2009]. The method estimates ``θ = \\frac{σ}{k}``

# Keywords

  - `sorted=false`: whether `x` is already sorted
  - `min_points=30`: default number of points to use to compute estimate.
    Instead of `min_points=20` as recommended by [^ZhangStephens2009], we use 30 points
    as in the loo software. [^ZhangStephens2009] notes that the estimator is not sensitive
    to this choice.
  - `adjust_prior=true`: whether to apply the weakly informative Gaussian prior on `k`
    suggested by [^VehtariSimpson2021] to reduce variance in the estimate of `k`.

[^ZhangStephens2009]: Zhang J & Stephens M A. (2009).
    A new and efficient estimation method for the generalized Pareto distribution,
    Technometrics, 51:3, 316-325,
    DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
function fit(::Type{<:GeneralizedPareto}, x; sorted=false, min_points=30, adjust_prior=true)
    x = sorted ? x : sort(x)
    n = length(x)
    m = min_points + floor(Int, sqrt(n))
    θ_hat = fit_θ(x, m)
    k_hat = fit_k(x, θ_hat)
    σ_hat = fit_σ(θ_hat, k_hat)
    # NOTE: the paper is ambiguous whether the adjustment is applied to k_hat
    # before or after computing σ_hat. From private discussion with Aki Vehtari,
    # adjusting afterwards produces better results.
    k_hat = adjust_prior ? prior_adjust_k(k_hat, n) : k_hat
    return GeneralizedPareto(σ_hat, k_hat)
end

# estimate θ̂ = ∫θp(θ|x)dθ using quadrature over m grid points
# uniformly sampled over the empirical prior
function fit_θ(x, m)
    T = float(eltype(x))
    n = length(x)

    # construct empirical prior on y = 1/x[n] - θ
    x_star = quartile(x, 1)
    σ_star = inv(6 * x_star)
    k_star = 1//2
    y_prior = GeneralizedPareto(σ_star, k_star)

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
function fit_k(x, θ_hat)
    nθ_hat = -θ_hat
    return mean(xᵢ -> log1p(nθ_hat * xᵢ), x)
end
fit_σ(θ_hat, k_hat) = -k_hat / θ_hat

# compute likelihood p(x|θ,k), estimating k from θ
function profile_loglikelihood(θ, x, n=length(x))
    # estimate k given θ
    nk_est = -fit_k(x, θ)
    return n * (log(θ / nk_est) + nk_est - 1)
end

# reduce variance of estimated k using a weakly informative Gaussian prior
# centered at `k_prior` corresponding to `nobs` observations
# Vehtari et al, Appendix C
prior_adjust_k(k, n, k_prior=1//2, nobs=10) = (n * k + nobs * k_prior) / (n + nobs)
