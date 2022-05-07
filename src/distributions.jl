function Base.convert(
    ::Type{Distributions.GeneralizedPareto{T}}, d::GeneralizedPareto{S}
) where {T<:Real,S<:Real}
    return Distributions.GeneralizedPareto{T}(T(d.μ), T(d.σ), T(d.k))
end
function Base.convert(
    ::Type{Distributions.GeneralizedPareto}, d::GeneralizedPareto{T}
) where {T<:Real}
    return convert(Distributions.GeneralizedPareto{T}, d)
end
