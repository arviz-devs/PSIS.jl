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

The fit is performed using the Empirical Bayes method of [^ZhangStephens2009][^Zhang2010].

# Keywords

  - `sorted::Bool=issorted(x)`: If `true`, `x` is assumed to be sorted. If `false`, a sorted
    copy of `x` is made.
  - `improved::Bool=true`: If `true`, use the adaptive empirical prior of [^Zhang2010].
    If `false`, use the simpler prior of [^ZhangStephens2009].
  - `min_points::Int=30`: The minimum number of quadrature points to use when estimating the
    posterior mean of ``\\theta = \\frac{\\xi}{\\sigma}``.

[^ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009)
    A New and Efficient Estimation Method for the Generalized Pareto Distribution,
    Technometrics, 51:3, 316-325,
    DOI: [10.1198/tech.2009.08017](https://doi.org/10.1198/tech.2009.08017)
[^Zhang2010]: Jin Zhang (2010) Improving on Estimation for the Generalized Pareto Distribution,
    Technometrics, 52:3, 335-339,
    DOI: [10.1198/TECH.2010.09206](https://doi.org/10.1198/TECH.2010.09206)
"""
fit_gpd(x::AbstractArray; kwargs...) = fit_gpd_empiricalbayes(x; kwargs...)

# Note: our k is ZhangStephens2009's -k, and our θ is ZhangStephens2009's -θ

function fit_gpd_empiricalbayes(
    x::AbstractArray;
    μ=zero(eltype(x)),
    sorted::Bool=issorted(vec(x)),
    improved::Bool=true,
    min_points::Int=30,
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
    θ_hat = _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points, improved)
    # estimate remaining parameters using MLE
    return _fit_gpd_mle_given_mu_theta(xsorted, μ, θ_hat)
end

# estimate θ̂ = ∫θp(θ|x,μ)dθ, i.e. the posterior mean using quadrature over grid
# of minimum length `min_points + floor(sqrt(length(x)))` uniformly sampled over an
# empirical prior
function _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points, improved)
    T = Base.promote_eltype(xsorted, μ)
    n = length(xsorted)

    # empirical prior on y = r/(xmax-μ) + θ
    θ_prior = if improved
        _gpd_empirical_prior_improved(μ, xsorted, n)
    else
        _gpd_empirical_prior(μ, xsorted, n)
    end

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

# Zhang, 2010
function _gpd_empirical_prior_improved(μ, xsorted, n=length(xsorted))
    xmax = xsorted[n]
    μ_star = (n - 1) / ((n + 1) * (μ - xmax))
    p = (3//10, 2//5, 1//2, 3//5, 7//10, 4//5, 9//10)  # 0.3:0.1:0.9
    q1 = (7, 6, 5, 4, 3, 2, 1)  # 10 .* (1 .- p)
    q2 = (91, 84, 75, 64, 51, 36, 19)  # 100 .* (1 .- p .^ 2)
    # q1/10- and q2/100- quantiles of xsorted without interpolation,
    # i.e. the α-quantile of x without interpolation is x[max(1, floor(Int, α * n + 1/2))]
    twon = 2 * n
    x1mp = map(qi -> xsorted[max(1, fld(qi * twon + 1, 20))], q1)
    x1mp2 = map(qi -> xsorted[max(1, fld(qi * twon + 1, 200))], q2)
    expkp = @. (x1mp2 - x1mp) / (x1mp - μ)
    σp = @. log(p, expkp) * (x1mp - μ) / (1 - expkp)
    σ_star = inv(2 * Statistics.median(σp))
    k_star = 1
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
