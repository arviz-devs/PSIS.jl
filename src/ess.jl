"""
    ess_is(weights; reff=1)

Estimate effective sample size (ESS) for importance sampling over the sample dimensions.

Given normalized weights ``w_{1:n}``, the ESS is estimated using the L2-norm of the weights:

```math
\\mathrm{ESS}(w_{1:n}) = \\frac{r_{\\mathrm{eff}}}{\\sum_{i=1}^n w_i^2}
```

where ``r_{\\mathrm{eff}}`` is the relative efficiency of the `log_weights`.

    ess_is(result::PSISResult; bad_shape_nan=true)

Estimate ESS for Pareto-smoothed importance sampling.

!!! note

    ESS estimates for Pareto shape values ``k > k_\\mathrm{threshold}``, which are
    unreliable and misleadingly high, are set to `NaN`. To avoid this, set
    `bad_shape_nan=false`.
"""
ess_is

function ess_is(r::PSISResult; bad_shape_nan::Bool=true)
    log_weights = r.log_weights
    if r.normalized
        weights = exp.(log_weights)
    else
        weights = LogExpFunctions.softmax(log_weights; dims=_sample_dims(log_weights))
    end
    ess = ess_is(weights; reff=r.reff)
    diagnostics = r.diagnostics
    khat = diagnostics.pareto_shape
    khat_thresh = diagnostics.pareto_shape_threshold
    return _apply_nan(ess, khat; khat_thresh, bad_shape_nan=bad_shape_nan)
end
function ess_is(weights; reff=1)
    dims = _sample_dims(weights)
    return reff ./ dropdims(sum(abs2, weights; dims=dims); dims=dims)
end

function _apply_nan(ess::Real, khat::Real; khat_thresh::Real, bad_shape_nan)
    bad_shape_nan || return ess
    (isnan(khat) || khat > khat_thresh) && return oftype(ess, NaN)
    return ess
end
function _apply_nan(ess::AbstractArray, khat::AbstractArray; kwargs...)
    return map(ess, khat) do essᵢ, khatᵢ
        return _apply_nan(essᵢ, khatᵢ; kwargs...)
    end
end
