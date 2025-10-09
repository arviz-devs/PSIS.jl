# utilities for computing properties or proxies of an expectand

_elementwise_transform(f::Base.Fix1{typeof(Statistics.mean)}) = f.x
_elementwise_transform(::Any) = identity

_max_moment_required(::Base.Fix1{typeof(Statistics.mean)}) = 1
_max_moment_required(::typeof(Statistics.mean)) = 1
_max_moment_required(::typeof(Statistics.var)) = 2
_max_moment_required(::typeof(Statistics.std)) = 2
_max_moment_required(::Base.Fix2{typeof(Statistics.quantile),<:Real}) = 0
_max_moment_required(::typeof(Statistics.median)) = 0

_requires_moments(f) = _max_moment_required(f) > 0

function _check_requires_moments(kind)
    _requires_moments(kind) && return nothing
    throw(
        ArgumentError("kind=$kind requires no moments. Pareto diagnostics are not useful.")
    )
end

# Compute an expectand `z` such that E[zr] requires the same number of moments as E[xr]
@inline function _expectand_proxy(f, x, r, is_x_log, is_r_log)
    fi = _elementwise_transform(f)
    p = _max_moment_required(f)
    if !is_x_log
        if !is_r_log
            return fi.(x) .^ p .* r
        else
            # scale ratios to maximum of 1 to avoid under/overflow
            return (fi.(x) .* exp.((r .- maximum(r; dims=_sample_dims(r))) ./ p)) .^ p
        end
    elseif fi === identity
        log_z = if is_r_log
            p .* x .+ r
        else
            p .* x .+ log.(r)
        end
        # scale to maximum of 1 to avoid overflow
        return exp.(log_z .- maximum(log_z; dims=_sample_dims(log_z)))
    else
        throw(
            ArgumentError(
                "cannot compute expectand proxy from log with non-identity transform"
            ),
        )
    end
end
