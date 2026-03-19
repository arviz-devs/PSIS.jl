function pareto_diagnose(
    x::AbstractVecOrMat;
    r_eff::Real=1,
    is_log_scale::Bool=false,
    tail::Symbol=(is_log_scale ? :right : :both),
)
    x_work = similar(x)
    copyto!(x_work, x)
    khat = _pareto_diagnose!(x_work; r_eff, is_log_scale, tail, smooth=false)
    n = length(x)
    return (;
        khat,
        min_ss=minimum_sample_size(khat),
        khat_threshold=oftype(khat, khat_threshold(n)),
        convergence_rate=convergence_rate(khat, n),
    )
end

function minimum_sample_size(k::Real)
    min_ss = exp10(inv(1 - max(0, k)))
    k < 1 && return min_ss
    return oftype(min_ss, Inf)
end

khat_threshold(ndraws::Integer) = 1 - inv(log10(ndraws))

function convergence_rate(k::Real, ndraws::Integer)
    T = typeof(one(k) / ndraws^one(k))
    k < 0 && return one(T)
    k == 1//2 && return T(1 - one(k) / log10(T(ndraws))) / 2
    k > 1 && return zero(T)
    rate = 2 - 2k + one(T) / (ndraws - 1) - (2k - 1) / (ndraws^(2k - 1) - 1)
    return T(max(0, rate))
end

function _pareto_diagnose!(
    x::AbstractVecOrMat; r_eff::Real, is_log_scale::Bool, tail::Symbol, smooth::Bool
)
    n = length(x)
    # get the tail length
    tail_length = _ps_tail_length(tail, n, r_eff)
    khat = _pareto_diagnose_tail!(vec(x), tail, n, tail_length, is_log_scale, smooth)
    return khat
end

function _pareto_diagnose_tail!(x, tail, length, tail_length, is_log_scale, smooth)
    is_log_scale &&
        !(tail === :right) &&
        throw(ArgumentError("Cannot diagnose $(tail) tail with log-scale data"))

    if tail === :right
        return _pareto_diagnose_tail_right!(x, length, tail_length, is_log_scale, smooth)
    elseif tail === :left
        return _pareto_diagnose_tail_left!(x, length, tail_length, smooth)
    elseif tail === :both
        return _pareto_diagnose_tail_both!(x, length, tail_length, smooth)
    else
        throw(ArgumentError("Invalid tail: $tail"))
    end
end

function _pareto_diagnose_tail_right!(x, length, tail_length, is_log_scale, smooth)
    perm = partialsortperm(x, (length - tail_length):length)
    cutoff = x[perm[1]]
    tail_inds = view(perm, 2:(tail_length + 1))
    x_tail = view(x, tail_inds)
    _, tail_dist = _psis_tail_right!(x_tail, cutoff; smooth)
    return tail_dist.k
end

function _pareto_diagnose_tail_left!(x, length, tail_length, smooth)
    perm = partialsortperm(x, (length - tail_length):length; rev=true)
    cutoff = x[perm[1]]
    tail_inds = view(perm, 2:(tail_length + 1))
    x_tail = view(x, tail_inds)
    @. x_tail = -x_tail
    _, tail_dist = _psis_tail_right!(x_tail, -cutoff; smooth)
    smooth && (@. x_tail = -x_tail)
    return tail_dist.k
end

function _pareto_diagnose_tail_both!(x, length, tail_length, smooth)
    perm = sortperm(x)
    cutoff = x[perm[length - tail_length]]
    tail_inds = view(perm, (length - tail_length + 1):length)
    x_tail = view(x, tail_inds)
    _, tail_dist = _psis_tail_right!(x_tail, cutoff; smooth)
    khat_right = tail_dist.k
    cutoff = -x[perm[tail_length + 1]]
    tail_inds = view(perm, tail_length:-1:1)
    x_tail = view(x, tail_inds)
    @. x_tail = -x_tail
    _, tail_dist = _psis_tail_right!(x_tail, cutoff; smooth)
    smooth && (@. x_tail = -x_tail)
    khat_left = tail_dist.k
    return max(khat_right, khat_left)
end

function _ps_tail_length(tail::Symbol, length, r_eff)
    max_length = cld(length, 5)
    if (isfinite(r_eff) && r_eff > 0)
        min_length = ceil(Int, 3 * sqrt(length / r_eff))
    else
        min_length = max_length
    end
    tail_length = min(max_length, min_length)
    if tail === :both
        tail_length = min(tail_length, length ÷ 2)
    end
    return tail_length
end
