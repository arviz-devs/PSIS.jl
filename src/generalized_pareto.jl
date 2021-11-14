
#
# MLE
#

"""
    GeneralizedParetoKnownMuTheta(μ, θ)

Represents a [`GeneralizedPareto`](@ref) where ``\\mu`` and ``\\theta=\\frac{\\xi}{\\sigma}`` are known.
"""
struct GeneralizedParetoKnownMuTheta{T} <: Distributions.IncompleteDistribution
    μ::T
    θ::T
end
GeneralizedParetoKnownMuTheta(μ, θ) = GeneralizedParetoKnownMuTheta(Base.promote(μ, θ)...)

struct GeneralizedParetoKnownMuThetaStats{T} <: Distributions.SufficientStats
    μ::T  # known mean
    θ::T  # known theta
    ξ::T  # known shape
end
function GeneralizedParetoKnownMuThetaStats(μ, θ, ξ)
    return GeneralizedParetoKnownMuThetaStats(Base.promote(μ, θ, ξ)...)
end

function Distributions.suffstats(d::GeneralizedParetoKnownMuTheta, x::AbstractArray)
    μ = d.μ
    θ = d.θ
    ξ = mean(xi -> log1p(θ * (xi - μ)), x) # mle estimate of ξ
    return GeneralizedParetoKnownMuThetaStats(μ, θ, ξ)
end

function Distributions.fit_mle(g::GeneralizedParetoKnownMuTheta, x::AbstractArray)
    ss = Distributions.suffstats(g, x)
    return Distributions.fit_mle(g, ss)
end
function Distributions.fit_mle(
    d::GeneralizedParetoKnownMuTheta, ss::GeneralizedParetoKnownMuThetaStats
)
    ξ = ss.ξ
    return Distributions.GeneralizedPareto(d.μ, ξ / d.θ, ξ)
end

#
# empirical bayes
#

"""
    GeneralizedParetoKnownMu(μ)

Represents a [`GeneralizedPareto`](@ref) where ``\\mu`` is known.
"""
struct GeneralizedParetoKnownMu{T} <: Distributions.IncompleteDistribution
    μ::T
end

"""
    fit(g::GeneralizedParetoKnownMu, x; kwargs...)

Fit a [`GeneralizedPareto`](@ref) with known location `μ` to the data `x`.

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
function StatsBase.fit(g::GeneralizedParetoKnownMu, x::AbstractArray; kwargs...)
    return fit_empiricalbayes(g, x; kwargs...)
end

# Note: our ξ is ZhangStephens2009's -k, and our θ is ZhangStephens2009's -θ

function fit_empiricalbayes(
    g::GeneralizedParetoKnownMu,
    x::AbstractArray;
    sorted::Bool=issorted(vec(x)),
    improved::Bool=true,
    min_points::Int=30,
)
    μ = g.μ
    T = Base.promote_eltype(x, μ)
    # fitting is faster when the data are sorted
    xsorted = sorted ? vec(x) : sort(vec(x))
    xmin, xmax = @inbounds xsorted[1], xsorted[end]
    if xmin ≈ xmax
        # support is nearly a point. solution is not unique; any solution satisfying the
        # constraints σ/ξ ≈ 0 and ξ < 0 is acceptable. we choose the ξ = -1 solution, i.e.
        # the uniform distribution
        return DistributionsGeneralizedPareto(μ, max(eps(zero(T)), xmax - μ), -one(μ))
    end
    # estimate θ using empirical bayes
    θ_hat = _fit_gpd_θ_empirical_bayes(μ, xsorted, min_points, improved)
    # estimate remaining parameters using MLE
    return Distributions.fit_mle(GeneralizedParetoKnownMuTheta(μ, θ_hat), xsorted)
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
    θ = quantile.(Ref(θ_prior), p)

    # estimate mean θ over the quadrature points
    # with weights as the normalized profile likelihood 
    lθ = _gpd_profile_loglikelihood.(μ, θ, Ref(xsorted), n)
    lθ_norm = logsumexp(lθ)
    θ_hat = @inbounds sum(1:npoints) do j
        wⱼ = exp(lθ[j] - lθ_norm)
        return θ[j] * wⱼ
    end

    return θ_hat
end

# Zhang & Stephens, 2009
function _gpd_empirical_prior(μ, xsorted, n=length(x))
    xmax = xsorted[n]
    μ_star = -inv(xmax - μ)
    x_25 = xsorted[fld(n + 2, 4)]
    σ_star = inv(6 * (x_25 - μ))
    ξ_star = 1//2
    return Distributions.GeneralizedPareto(μ_star, σ_star, ξ_star)
end

# Zhang, 2010
function _gpd_empirical_prior_improved(μ, xsorted, n=length(x))
    xmax = xsorted[n]
    μ_star = -inv(xmax - μ) * ((n - 1)//(n + 1))
    p = (3:9) ./ oftype(μ_star, 10)
    q = [1 .- p; 1 .- p .^ 2]
    xquantiles = if VERSION ≥ v"1.5.0"
        quantile(xsorted, q; sorted=true, alpha=0, beta=1)
    else
        quantile(xsorted, q; sorted=true)
    end
    x1mp, x1mp2 = @views xquantiles[1:7], xquantiles[8:14]
    expkp = @. (x1mp2 - x1mp) / (x1mp - μ)
    σp = @. log(p, expkp) * (x1mp - μ) / (1 - expkp)
    σ_star = inv(2 * median(σp))
    ξ_star = 1
    return Distributions.GeneralizedPareto(μ_star, σ_star, ξ_star)
end

# compute log joint likelihood p(x|μ,θ), with ξ the MLE given θ and x
function _gpd_profile_loglikelihood(μ, θ, x, n=length(x))
    g = GeneralizedParetoKnownMuTheta(μ, θ)
    d = Distributions.fit_mle(g, x)
    return -n * (log(d.σ) + d.ξ + 1)
end

function prior_adjust_shape(d::Distributions.GeneralizedPareto, n, ξ_prior=1//2, nobs=10)
    ξ = (n * d.ξ + nobs * ξ_prior) / (n + nobs)
    return Distributions.GeneralizedPareto(d.μ, d.σ, ξ)
end
