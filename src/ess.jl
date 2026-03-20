"""
    ess_is(w; r_eff=1)

Estimate effective sample size (ESS) for importance sampling over the sample dimensions.

Given normalized (log-)weights ``w_{1:n}``, the ESS is estimated using the L2-norm of the
weights:

```math
\\mathrm{ESS}(w_{1:n}) = \\frac{r_{\\mathrm{eff}}}{\\sum_{i=1}^n w_i^2}
```

where ``r_{\\mathrm{eff}}`` is the relative efficiency of the (log-)weights.
"""
function ess_is(log_weights::AbstractArray, r_eff)
    dims = _param_dims(log_weights)
    return ess_is.(eachslice(log_weights; dims), r_eff)
end
function ess_is(log_weights::AbstractVecOrMat, r_eff)
    lw_max = maximum(log_weights)
    T = typeof(lw_max)
    s1, s2 = reduce(log_weights; init=(zero(T), zero(T))) do (s1, s2), lw
        d = lw - lw_max
        return (s1 + exp(d), s2 + exp(2 * d))
    end
    return T(only(r_eff * s1^2 / s2))
end
