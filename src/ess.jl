"""
    ess_is(weights; reff=1)

Estimate effective sample size (ESS) for importance sampling over the sample dimensions.

Given normalized weights ``w_{1:n}``, the ESS is estimated using the L2-norm of the weights:

```math
\\mathrm{ESS}(w_{1:n}) = \\frac{r_{\\mathrm{eff}}}{\\sum_{i=1}^n w_i^2}
```

where ``r_{\\mathrm{eff}}`` is the relative efficiency of the `log_weights`.

    ess_is(result::PSISResult; bad_shape_missing=true)

Estimate ESS for Pareto-smoothed importance sampling.

!!! note

    ESS estimates for Pareto shape values ``k > 0.7``, which are unreliable and misleadingly
    high, are set to `missing`. To avoid this, set `bad_shape_missing=false`.
"""
ess_is

function ess_is(r::PSISResult; bad_shape_missing::Bool=true)
    neff = ess_is(r.weights; reff=r.reff)
    return _apply_missing(neff, r.tail_dist; bad_shape_missing=bad_shape_missing)
end
function ess_is(weights; reff=1)
    dims = sample_dims(weights)
    neff = broadcast_last_dims(/, reff, sum(abs2, weights; dims=dims))
    return dropdims(neff; dims=dims)
end
ess_is(weights::AbstractVector; reff::Real=1) = reff / sum(abs2, weights)

function _apply_missing(neff, dist; bad_shape_missing)
    return bad_shape_missing && pareto_shape(dist) > 0.7 ? missing : neff
end
_apply_missing(neff, ::Missing; kwargs...) = missing
function _apply_missing(ess::AbstractArray, tail_dist::AbstractArray; kwargs...)
    return map(ess, tail_dist) do essᵢ, tail_distᵢ
        return _apply_missing(essᵢ, tail_distᵢ; kwargs...)
    end
end
