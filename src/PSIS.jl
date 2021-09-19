module PSIS

using Statistics: mean
using LinearAlgebra: dot

export psis

include("generalized_pareto.jl")

function psis(logr, r_eff)
    T = float(eltype(logr))
    S = length(logr)
    logw = copyto!(similar(logr, T), logr)
    k_hat = T(Inf)

    M = tail_length(r_eff, S)
    if M < 5
        @warn "Insufficient tail draws to fit the generalized Pareto distribution."
        return logw, k_hat
    end

    perm = sortperm(logw)
    logw_max = logw[last(perm)]
    icut = S - M
    tail_range = (icut + 1):S

    logw_tail = @views logw[perm[tail_range]]
    logw_tail .-= logw_max
    logu = logw[perm[icut]] - logw_max
    
    _, k_hat = psis_tail!(logw_tail, logu, M; sorted=true)
    logw_tail .+= logw_max

    k_hat > 0.7 && @warn "Pareto k statistic exceeded 0.7. Resulting importance sampling estimates are likely to be unstable."

    return logw, T(k_hat)
end

tail_length(r_eff, S) = min(cld(S, 5), ceil(Int, 3 * sqrt(S / r_eff)))

function psis_tail!(logw, logu, M=length(logw); sorted=false)
    sorted || sort!(logw)
    T = eltype(logw)
    u = exp(logu)
    w = exp.(logw) .- u
    d_hat = fit(GeneralizedPareto, w)
    k_hat = T(d_hat.k)
    if isfinite(k_hat)
        z = 1:M
        p = (z .- T(1//2)) ./ M
        logw .= min.(log.(quantile.(Ref(d_hat), p) .+ u), zero(T))
    end
    return logw, k_hat
end

end
