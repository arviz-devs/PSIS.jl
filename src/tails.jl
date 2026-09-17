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
    return floor(Int, min(n / 5, 3 * sqrt(n / r_eff)))
end

function _fit_tail(x::AbstractVector, m::Integer; tail::Symbol, is_log_scale::Bool)
    T = float(eltype(x))
    n = length(x)
    5 <= m < n || return _failed_tail(T, :insufficient)
    if is_log_scale ? any(v -> isnan(v) || v == Inf, x) : any(!isfinite, x)
        return _failed_tail(T, :nonfinite)
    end
    left = tail === :left
    perm = partialsortperm(x, (n - m):n; rev=left)
    indices = @view perm[2:end]
    values = @view x[indices]
    all(isfinite, values) || return _failed_tail(T, :nonfinite)
    first(values) == last(values) && return _failed_tail(T, :constant)
    sign = left ? -one(T) : one(T)
    extreme = T(last(values))
    cutoff_raw = T(x[first(perm)])
    if is_log_scale
        shift = extreme
        scale = one(T)
        cutoff = exp(cutoff_raw - shift)
        # exp(a) - exp(b), evaluated without cancellation when a ≈ b.
        excesses = @. exp(values - shift) * -expm1(cutoff_raw - values)
    else
        shift = zero(T)
        scale = max(abs(extreme), abs(cutoff_raw))
        cutoff = sign * cutoff_raw / scale
        excesses = @. sign * values / scale - cutoff
    end
    dist = fit_gpd(excesses; prior_adjusted=true, sorted=true)
    status = isfinite(dist.k) && isfinite(dist.σ) && dist.σ > 0 ? :ok : :failed
    status === :ok || return _failed_tail(T, status)
    return (; indices, dist, cutoff, scale, shift, sign, extreme, status)
end
