# Internal fit metadata are shared by diagnosis and smoothing. The distribution
# and cutoff use scaled, right-oriented coordinates; indices preserve draw order.
function _failed_tail(::Type{T}, status) where {T}
    nan = T(NaN)
    return (;
        indices=Int[],
        dist=GeneralizedPareto(zero(T), nan, nan),
        cutoff=nan,
        scale=one(T),
        shift=zero(T),
        sign=one(T),
        extreme=nan,
        status,
    )
end

function _tail_length(n::Integer, r_eff::Real)
    isfinite(r_eff) && r_eff > 0 ||
        throw(ArgumentError("r_eff must be positive and finite"))
    # Take the minimum before converting to Int, including when n/r_eff overflows.
    return floor(Int, min(n / 5, 3 * sqrt(n / r_eff)))
end

function _fit_tail(x::AbstractVector, m::Integer; tail::Symbol, is_log_scale::Bool)
    T = float(eltype(x))
    5 <= m < length(x) || return _failed_tail(T, :insufficient)
    invalid = is_log_scale ? any(v -> isnan(v) || v == Inf, x) : any(!isfinite, x)
    invalid && return _failed_tail(T, :nonfinite)
    left = tail === :left
    perm = partialsortperm(x, (length(x) - m):length(x); rev=left)
    indices = perm[2:end]
    values = x[indices]
    all(isfinite, values) || return _failed_tail(T, :nonfinite)
    first(values) == last(values) && return _failed_tail(T, :constant)
    sign = left ? -one(T) : one(T)
    extreme = T(last(values))
    cutoff_raw = T(x[first(perm)])
    shift = is_log_scale ? extreme : zero(T)
    scale = is_log_scale ? one(T) : max(abs(extreme), abs(cutoff_raw))
    cutoff = is_log_scale ? exp(cutoff_raw - shift) : sign * cutoff_raw / scale
    excesses = if is_log_scale
        # exp(a) - exp(b), evaluated without cancellation when a ≈ b.
        T[exp(v - shift) * (-expm1(cutoff_raw - v)) for v in values]
    else
        T[sign * v / scale - cutoff for v in values]
    end
    dist = fit_gpd(excesses; prior_adjusted=true, sorted=true)
    status = isfinite(dist.k) && isfinite(dist.σ) && dist.σ > 0 ? :ok : :failed
    status === :ok || return _failed_tail(T, status)
    return (; indices, dist, cutoff, scale, shift, sign, extreme, status)
end
