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

    ESS estimates for Pareto shape values ``k > 0.7``, which are unreliable and misleadingly
    high, are set to `NaN`. To avoid this, set `bad_shape_nan=false`.
"""
ess_is

function ess_is(r::PSISResult; bad_shape_nan::Bool=true)
    neff = ess_is(r.weights; reff=r.reff)
    return _apply_nan(neff, r.tail_dist; bad_shape_nan=bad_shape_nan)
end
function ess_is(weights; reff=1)
    dims = _sample_dims(weights)
    return reff ./ dropdims(sum(abs2, weights; dims=dims); dims=dims)
end

function _apply_nan(neff, dist; bad_shape_nan)
    bad_shape_nan || return neff
    k = pareto_shape(dist)
    (isnan(k) || k > 0.7) && return oftype(neff, NaN)
    return neff
end
function _apply_nan(ess::AbstractArray, tail_dist::AbstractArray; kwargs...)
    return map(ess, tail_dist) do essᵢ, tail_distᵢ
        return _apply_nan(essᵢ, tail_distᵢ; kwargs...)
    end
end
