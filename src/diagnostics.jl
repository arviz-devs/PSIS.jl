"""
    pareto_shape_threshold(sample_size::Real) -> Real

Given the `sample_size`, compute the Pareto shape ``k`` threshold needed for a reliable
Pareto-smoothed estimate (i.e. to have small probability of large error).
"""
pareto_shape_threshold(sample_size::Real) = 1 - inv(log10(sample_size))

"""
    min_sample_size(pareto_shape::Real) -> Real
    min_sample_size(pareto_shape::AbstractArray) -> AbstractArray

Given the Pareto shape values ``k``, compute the minimum sample size needed for a reliable
Pareto-smoothed estimate (i.e. to have small probability of large error).
"""
function min_sample_size end
min_sample_size(pareto_shape::Real) = exp10(inv(1 - max(0, pareto_shape)))
min_sample_size(pareto_shape::AbstractArray) = map(min_sample_size, pareto_shape)

"""
    convergence_rate(pareto_shape::Real, sample_size::Real) -> Real
    convergence_rate(pareto_shape::AbstractArray, sample_size::Real) -> AbstractArray

Given `sample_size` and Pareto shape values ``k``, compute the relative convergence rate of
the RMSE of the Pareto-smoothed estimate.
"""
function convergence_rate end
function convergence_rate(k::AbstractArray{<:Real}, S::Real)
    return convergence_rate.(k, S)
end
function convergence_rate(k::Real, S::Real)
    T = typeof((one(S) * 1^zero(k) * oneunit(k)) / (one(S) * 1^zero(k)))
    k < 0 && return oneunit(T)
    k > 1 && return zero(T)
    k == 1//2 && return T(1 - inv(log(S)))
    return T(
        max(
            0,
            (2 * (k - 1) * S^(2k) - (2k - 1) * S^(2k - 1) + S) /
            ((S - 1) * (1 - S^(2k - 1))),
        ),
    )
end
