"""
    pareto_min_ss(x::AbstractArray; kwargs...)
    pareto_min_ss(k::Real)

Estimate the minimum sample size for a reliable Pareto-smoothed estimate.

For observations `x`, fit k-hat using the arguments and keywords of
[`pareto_khat`](@ref). For an already estimated shape `k`, evaluate the diagnostic
directly. Arrays always represent observations; broadcast `pareto_min_ss.(khats)`
for an array of fitted shapes.

# Interpretation

Returns `10^(1 / (1 - max(0, k)))` for `k < 1`, and `Inf` otherwise;
`NaN` propagates. The lower bound is ten draws. Large values suggest that many
more draws are needed; `Inf` indicates that the heuristic gives no finite
sample-size recommendation. This is approximate guidance for a smoothed estimate,
not a guarantee of accuracy or a test of MCMC convergence.

See Section 3.2.3 of [VehtariSimpson2021](@citet) and
[`pareto_khat_threshold`](@ref) for the inverse relationship.
"""
function pareto_min_ss(k::Real)
    T = float(typeof(k))
    k > 1 && return T(Inf)
    return exp10(inv(1 - max(zero(T), k)))
end

"""
    pareto_khat_threshold(x::AbstractArray)
    pareto_khat_threshold(n::Real)

Compute the sample-size-dependent k-hat threshold.

For observations `x`, pool the first two dimensions (draws and chains), without
fitting a tail. For a supplied sample size `n`, require a finite value greater
than one. The observation method returns `NaN` for fewer than two draws.
Broadcast `pareto_khat_threshold.(ns)` for several supplied sample sizes.

# Interpretation

Returns `1 - 1/log10(n)`. Exceeding this threshold suggests insufficient draws
for a reliable Pareto-smoothed estimate; see [`pareto_min_ss`](@ref).
The rule of thumb is intended for `n ≥ 10`; the formula is also evaluated for
`1 < n < 10`, where it is negative.

This threshold is uncapped. Warnings and summary displays use
`min(0.7, pareto_khat_threshold(n))`, accounting also for bias at high k-hat.
Passing either threshold is a heuristic, not a guarantee of accuracy.
See Section 3.2.4 of [VehtariSimpson2021](@citet).
"""
function pareto_khat_threshold(n::Real)
    _check_sample_size(n)
    return 1 - inv(log10(n))
end

"""
    pareto_convergence_rate(x::AbstractArray; kwargs...)
    pareto_convergence_rate(k::Real, n::Real)

Estimate the relative RMSE convergence rate of a Pareto-smoothed estimate.

For observations `x`, fit k-hat using the arguments and keywords of
[`pareto_khat`](@ref), pooling draws and chains for the sample size.
Alternatively supply an estimated shape `k` and finite sample size `n > 1`.
Broadcast `pareto_convergence_rate.(khats, n)` for an array of fitted shapes.
The observation method returns `NaN` for fewer than two draws.

# Interpretation

The returned rate `α` describes local RMSE scaling approximately as `n^(-α/2)`.
One is the usual Monte Carlo rate; smaller positive values imply slower error
reduction. For `α > 0`, halving RMSE locally requires roughly `4^(1/α)` times
as many draws. This prediction is approximate and depends on the tail model;
it does not diagnose MCMC mixing.

Uses the smooth approximation in Appendix B of [VehtariSimpson2021](@citet), evaluated
stably near `k = 0.5`. Its continuous limit there is
`1 + 1/(n-1) - 1/log(n)`, retaining the finite-sample correction `1/(n-1)`
omitted from the paper's simplified special case. Rates are restricted to `[0, 1]`.
The implementation returns one for `k ≤ 0`, zero for `k ≥ 1`, and propagates
`NaN`. Zero gives no useful error-reduction prediction.
"""
function pareto_convergence_rate(k::Real, n::Real)
    _check_sample_size(n)
    T = float(promote_type(typeof(k), typeof(n)))
    k <= 0 && return one(T)
    k >= 1 && return zero(T)

    logn = log(T(n))
    a = 2 * T(k) - 1
    z = a * logn
    # a / expm1(z) → 1/log(n) as a → 0
    ratio = iszero(z) ? inv(logn) : a / expm1(z)
    rate = 1 + inv(T(n) - 1) - a - ratio
    return clamp(rate, 0, 1)
end

function _check_sample_size(n)
    isfinite(n) && n > 1 ||
        throw(ArgumentError("sample size must be finite and greater than 1"))
    return nothing
end
