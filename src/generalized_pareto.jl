# Note: These internal functions are here to avoid a dependency on Distributions.jl,
# which currently does not implement GPD fitting anyways. They are not methods of
# functions in Statistics/StatsBase

struct GeneralizedPareto{T}
    σ::T
    k::T
end
GeneralizedPareto(σ, k) = GeneralizedPareto(Base.promote(σ, k)...)

# Compute the `p`-quantile of the generalized Pareto distribution `d`.
@inline function quantile(d::GeneralizedPareto, p)
    k = d.k
    z = -log1p(-p)
    return iszero(k) ? d.σ * z : expm1(k * z) * (d.σ / k)
end

function fit(::Type{<:GeneralizedPareto}, x; sorted=false, min_points=30, adjust_prior=true)
    x = sorted ? x : sort(x)
    n = length(x)
    m = min_points + floor(Int, sqrt(n))
    θ_hat = estimate_θ(x, m)
    k_hat = estimate_k(x, θ_hat)
    σ_hat = estimate_σ(θ_hat, k_hat)
    # NOTE: the paper is ambiguous whether the adjustment is applied to k_hat
    # before or after computing σ_hat. From private discussion with Aki Vehtari,
    # adjusting afterware produces better results.
    k_hat = adjust_prior ? prior_adjust_k(k_hat, n) : k_hat
    return GeneralizedPareto(σ_hat, k_hat)
end

# estimate θ̂ = ∫ θ p(θ|x) dθ using quadrature over m grid points
# uniformly sampled over the empirical prior
function estimate_θ(x, m)
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

function estimate_k(x, θ_hat)
    nθ_hat = -θ_hat
    return mean(xᵢ -> log1p(nθ_hat * xᵢ), x)
end

estimate_σ(θ_hat, k_hat) = -k_hat / θ_hat

# compute likelihood p(x|θ,k), estimating k from θ
function profile_loglikelihood(θ, x, n=length(x))
    # estimate k given θ
    nk_est = -estimate_k(x, θ)
    return n * (log(θ / nk_est) + nk_est - 1)
end

# reduce variance of estimated k using a weakly informative Gaussian prior
# centered at `k_prior` corresponding to `nobs` observations
# Vehtari et al, Appendix C
prior_adjust_k(k, n, k_prior=1//2, nobs=10) = (n * k + nobs * k_prior) / (n + nobs)
