"""
    ess_is(weights; r_eff=1)

Estimate effective sample size (ESS) for importance sampling over the sample dimensions.

Given normalized weights ``w_{1:n}``, the ESS is estimated using the L2-norm of the weights:

```math
\\mathrm{ESS}(w_{1:n}) = \\frac{r_{\\mathrm{eff}}}{\\sum_{i=1}^n w_i^2}
```

where ``r_{\\mathrm{eff}}`` is the relative efficiency of the `log_weights`.

    ess_is(result::PSISResult; bad_shape_nan=true)

Estimate ESS for Pareto-smoothed importance sampling.

!!! note

    ESS estimates for Pareto shape values ``k > 0.7``, which are unreliable and misleadingly
    high, are set to `NaN`. To avoid this, set `bad_shape_nan=false`.
"""
ess_is

function ess_is(r::PSISResult; bad_shape_nan::Bool=true)
    weights = importance_weights(r.log_weights)
    neff = ess_is(weights; r_eff=r.r_eff)
    return _apply_nan(neff, r.pareto_shape; bad_shape_nan=bad_shape_nan)
end
function ess_is(weights; r_eff=1)
    dims = _sample_dims(weights)
    return r_eff ./ dropdims(sum(abs2, weights; dims=dims); dims=dims)
end

function _apply_nan(neff, k; bad_shape_nan)
    bad_shape_nan || return neff
    (isnan(k) || k > 0.7) && return oftype(neff, NaN)
    return neff
end
function _apply_nan(ess::AbstractArray, khat::AbstractArray; kwargs...)
    return map(ess, khat) do essᵢ, khatᵢ
        return _apply_nan(essᵢ, khatᵢ; kwargs...)
    end
end
