# Note: These internal functions are here to avoid a dependency on Distributions.jl,
# which currently does not implement GPD fitting anyways. They are not methods of
# functions in Statistics/StatsBase

"""
    GeneralizedPareto{T<:Real}

The generalized Pareto distribution.

This is equivalent to `Distributions.GeneralizedPareto` and can be converted to one with
`convert(Distributions.GeneralizedPareto, d)`.

# Constructor

    GeneralizedPareto(μ, σ, k)

Construct the generalized Pareto distribution (GPD) with location parameter ``μ``, scale
parameter ``σ`` and shape parameter ``k``.

!!! note

    The shape parameter ``k`` is equivalent to the commonly used shape parameter ``ξ``.
    This is the same parameterization used by [^VehtariSimpson2021] and is related to that
    used by [^ZhangStephens2009] as ``k \\mapsto -k``.
"""
struct GeneralizedPareto{T}
    μ::T
    σ::T
    k::T
end
GeneralizedPareto(μ, σ, k) = GeneralizedPareto(Base.promote(μ, σ, k)...)

function quantile(d::GeneralizedPareto{T}, p::Real) where {T<:Real}
    nlog1pp = -log1p(-p * one(T))
    k = d.k
    z = abs(k) < eps() ? nlog1pp : expm1(k * nlog1pp) / k
    return muladd(d.σ, z, d.μ)
end

#
# MLE
#

function _fit_gpd_mle_given_mu_theta(x::AbstractArray, μ, θ)
    k = Statistics.mean(xi -> log1p(θ * (xi - μ)), x) # mle estimate of k
    σ = k / θ
    return GeneralizedPareto(μ, σ, k)
end

#
# empirical bayes
#

"""
    fit_gpd(x; μ=0, kwargs...)

Fit a [`GeneralizedPareto`](@ref) with location `μ` to the data `x`.

The fit is performed using the Empirical Bayes method of [^ZhangStephens2009].

# Keywords

  - `prior_adjusted::Bool=true`, If `true`, a weakly informative Normal prior centered on
    ``\\frac{1}{2}`` is used for the shape ``k``.
  - `sorted::Bool=issorted(x)`: If `true`, `x` is assumed to be sorted. If `false`, a sorted
    copy of `x` is made.
  - `min_points::Int=30`: The minimum number of quadrature points to use when estimating the
    posterior mean of ``\\theta = \\frac{\\xi}{\\sigma}``.

[^ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009)
    A New and Efficient Estimation Method for the Generalized Pareto Distribution,
    Technometrics, 51:3, 316-325,
    DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
"""
function fit_gpd(x::AbstractArray; prior_adjusted::Bool=true, kwargs...)
    tail_dist = fit_gpd_empiricalbayes(x; kwargs...)
    if prior_adjusted && !_is_uniform(tail_dist)
        return prior_adjust_shape(tail_dist, length(x))
    else
        return tail_dist
    end
end

_is_uniform(d::GeneralizedPareto) = iszero(d.σ) && isone(-d.k)

# Note: our k is ZhangStephens2009's -k, and our θ is ZhangStephens2009's -θ

function fit_gpd_empiricalbayes(
    x::AbstractArray; μ=zero(eltype(x)), sorted::Bool=issorted(vec(x)), min_points::Int=30
)
    # fitting is faster when the data are sorted
    xsorted = sorted ? vec(x) : sort(vec(x))
    xmin = first(xsorted)
    xmax = last(xsorted)
    if xmin ≈ xmax
        # support is nearly a point. solution is not unique; any solution satisfying the
        # constraints σ/k ≈ 0 and k < 0 is acceptable. we choose the k = -1 solution, i.e.
        # the uniform distribution
        return GeneralizedPareto(μ, xmax - μ, -1)
    end
    # estimate θ using empirical bayes
    θ_hat = _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points)
    # estimate remaining parameters using MLE
    return _fit_gpd_mle_given_mu_theta(xsorted, μ, θ_hat)
end

# estimate θ̂ = ∫θp(θ|x,μ)dθ, i.e. the posterior mean using quadrature over grid
# of minimum length `min_points + floor(sqrt(length(x)))` uniformly sampled over an
# empirical prior
function _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points)
    T = Base.promote_eltype(xsorted, μ)
    n = length(xsorted)

    # empirical prior on y = r/(xmax-μ) + θ
    θ_prior = _gpd_empirical_prior(μ, xsorted, n)

    # quadrature points uniformly spaced on the quantiles of the θ prior
    npoints = min_points + floor(Int, sqrt(n))
    p = uniform_probabilities(T, npoints)
    θ = map(Base.Fix1(quantile, θ_prior), p)

    # estimate mean θ over the quadrature points
    # with weights as the normalized profile likelihood
    lθ = map(θ -> _gpd_profile_loglikelihood(μ, θ, xsorted, n), θ)
    lθ_norm = LogExpFunctions.logsumexp(lθ)
    θ_hat = @inbounds sum(1:npoints) do j
        wⱼ = exp(lθ[j] - lθ_norm)
        return θ[j] * wⱼ
    end

    return θ_hat
end

# Zhang & Stephens, 2009
function _gpd_empirical_prior(μ, xsorted, n=length(xsorted))
    xmax = xsorted[n]
    μ_star = -inv(xmax - μ)
    x_25 = xsorted[max(fld(n + 2, 4), 1)]
    σ_star = inv(6 * (x_25 - μ))
    k_star = 1//2
    return GeneralizedPareto(μ_star, σ_star, k_star)
end

# compute log joint likelihood p(x|μ,θ), with k the MLE given θ and x
function _gpd_profile_loglikelihood(μ, θ, x, n=length(x))
    d = _fit_gpd_mle_given_mu_theta(x, μ, θ)
    return -n * (log(d.σ) + d.k + 1)
end

function prior_adjust_shape(d::GeneralizedPareto, n, k_prior=1//2, nobs=10)
    k = (n * d.k + nobs * k_prior) / (n + nobs)
    return GeneralizedPareto(d.μ, d.σ, k)
end
