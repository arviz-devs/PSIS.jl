# Note: These internal functions are here to avoid a dependency on Distributions.jl,
# which currently does not implement GPD fitting anyways. They are not methods of
# functions in Statistics/StatsBase

struct GeneralizedPareto{T}
    σ::T
    k::T
end

@inline function quantile(d::GeneralizedPareto, p)
    k = d.k
    z = -log1p(-p)
    return iszero(k) ? d.σ * z : expm1(k * z) * (d.σ / k)
end

function fit(::Type{<:GeneralizedPareto}, x; sorted=false, min_points=30, adjust_prior=true)
    x = sorted ? x : sort(x)
    n = length(x)
    m = min_points + floor(Int, sqrt(n))
    θ_hat = estimate_θ(x, m, n)
    k_hat = estimate_k(x, θ_hat)
    σ_hat = estimate_σ(θ_hat, k_hat)
    # NOTE: the paper is ambiguous whether the adjustment is applied to k_hat
    # before or after computing σ_hat. loo adjusts afterwards while arviz
    # adjusts before. We follow the loo convention.
    k_hat = adjust_prior ? prior_adjust_k(k_hat, n) : k_hat
    return GeneralizedPareto(σ_hat, k_hat)
end

function estimate_θ(x, m, n=length(x))
    T = float(eltype(x))
    p = ((1:m) .- T(1//2)) ./ m
    @inbounds x_star = x[fld(n + 2, 4)]  # first quartile of x
    @inbounds θ = inv(x[n]) .+ (1 .- inv.(sqrt.(p))) ./ (3x_star)
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

prior_adjust_k(k, n) = (n * k + 5) / (n + 10)

function profile_loglikelihood(θ, x, n=length(x))
    nθ = -θ
    k = mean(xᵢ -> log1p(nθ * xᵢ), x)
    return n * (log(nθ / k) - k - 1)
end
