# utilities for specifying or retrieving tails

@enum Tails LeftTail RightTail BothTails
const TAIL_OPTIONS = (left=LeftTail, right=RightTail, both=BothTails)

_standardize_tails(tails::Tails) = tails
function _standardize_tails(tails::Symbol)
    if !haskey(TAIL_OPTIONS, tails)
        throw(
            ArgumentError("invalid tails: $tails. Valid values are $(keys(TAIL_OPTIONS)))")
        )
    end
    return TAIL_OPTIONS[tails]
end

function tail_length(reff, S)
    (isfinite(reff) && reff > 0 && S > 225) || return cld(S, 5)
    return ceil(Int, 3 * sqrt(S / reff))
end

function _tail_length(reff, S, tails::Tails)
    M = tail_length(reff, S)
    if tails === BothTails && M > fld(S, 2)
        M = Int(fld(S, 2))
    end
    return M
end

function _tail_and_cutoff(x::AbstractVector, M::Integer, tail::Tails)
    S = length(x)
    ind_offset = firstindex(x) - 1
    perm = partialsortperm(x, ind_offset .+ ((S - M):S); rev=tail === LeftTail)
    cutoff = x[first(perm)]
    tail_inds = @view perm[(firstindex(perm) + 1):end]
    return @views x[tail_inds], cutoff
end

function _shift_tail!(x_tail_shifted, x_tail, cutoff, tails::Tails)
    if tails === LeftTail
        @. x_tail_shifted = cutoff - x_tail
    else
        @. x_tail_shifted = x_tail - cutoff
    end
    return x_tail_shifted
end
