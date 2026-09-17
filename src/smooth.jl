"""
    pareto_smooth(x; r_eff=1, tail=:both, is_log_scale=false, tail_length=nothing, warn=false)

Return `(smoothed, diagnostics)`, replacing the selected tails of `x` with
fitted generalized Pareto quantiles. The original observations are unchanged;
the returned array preserves their order, shape and parameter axes. Values
outside the tails are unchanged, and each tail is capped at its observed extreme.
Generic samples are never normalized.

Keywords and diagnostic fields are those of [`pareto_khat`](@ref) and
[`pareto_diagnose`](@ref). Each tail is fitted once. An unfit tail is left
unchanged and its k-hat is `NaN`. Use [`psis`](@ref) for importance weights.
"""
function pareto_smooth(
    x::AbstractArray{<:Real};
    r_eff=1,
    is_log_scale::Bool=false,
    tail::Symbol=(is_log_scale ? :right : :both),
    tail_length::Union{Nothing,Integer}=nothing,
    warn::Bool=false,
)
    _check_pareto_inputs(x, r_eff, tail, is_log_scale, tail_length)
    smoothed = similar(x, float(eltype(x)))
    copyto!(smoothed, x)
    k = _map_parameters(smoothed, r_eff) do values, reff
        m = isnothing(tail_length) ? _tail_length(length(values), reff) : tail_length
        return _smooth_tails!(vec(values), m, tail, is_log_scale)
    end
    warn && _warn_diagnostics(k, _sample_size(x))
    return smoothed, _pareto_diagnostics(k, _sample_size(x))
end

function _smooth_tails!(x, m, tail, is_log_scale)
    if tail === :both
        left = _fit_tail(x, m; tail=:left, is_log_scale)
        right = _fit_tail(x, m; tail=:right, is_log_scale)
        _smooth_tail!(x, left, is_log_scale)
        _smooth_tail!(x, right, is_log_scale)
        return max(left.dist.k, right.dist.k)
    end
    fit = _fit_tail(x, m; tail, is_log_scale)
    _smooth_tail!(x, fit, is_log_scale)
    return fit.dist.k
end

function _smooth_tail!(x, fit, is_log_scale)
    fit.status === :ok || return x
    T = float(eltype(x))
    probabilities = uniform_probabilities(T, length(fit.indices))
    map!(view(x, fit.indices), probabilities) do p
        value = quantile(fit.dist, p) + fit.cutoff
        if is_log_scale
            min(log(value) + fit.shift, fit.extreme)
        else
            fit.sign * min(value * fit.scale, fit.sign * fit.extreme)
        end
    end
    return x
end
