"""
    ps_min_ss(k::Real)

Estimate the minimum sample size for a reliable Pareto-smoothed estimate from an
already estimated Pareto shape `k`: `10^(1 / (1 - max(0, k)))` for `k < 1`,
and `Inf` otherwise. This is a finite-sample heuristic, not a guarantee of accuracy.
"""
function ps_min_ss(k::Real)
    T = float(typeof(k))
    k > 1 && return T(Inf)
    return exp10(inv(1 - max(zero(T), k)))
end

"""
    ps_khat_threshold(n::Real)

Compute the sample-size-dependent threshold `1 - 1 / log10(n)` for `n > 1`.
For importance sampling, the operational reliability threshold is the smaller
of this value and `0.7`.
"""
function ps_khat_threshold(n::Real)
    _check_sample_size(n)
    return 1 - inv(log10(n))
end

"""
    ps_convergence_rate(k::Real, n::Real)

Estimate the relative convergence rate `α` of a Pareto-smoothed estimate from
its Pareto shape and sample size. Locally, RMSE scales approximately as `n^(-α/2)`;
the usual Monte Carlo rate corresponds to `α = 1`.

Uses the smooth approximation in Appendix B of Vehtari et al. (2024), evaluated
stably near `k = 0.5`. Its continuous limit there is
`1 + 1/(n-1) - 1/log(n)`, retaining the finite-sample correction `1/(n-1)`
omitted from the paper's simplified special case. Rates are restricted to `[0, 1]`.
"""
function ps_convergence_rate(k::Real, n::Real)
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
