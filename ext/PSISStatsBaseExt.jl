module PSISStatsBaseExt

using PSIS, StatsBase

PSIS._max_moment_required(::typeof(StatsBase.skewness)) = 3
PSIS._max_moment_required(::typeof(StatsBase.kurtosis)) = 4
PSIS._max_moment_required(f::Base.Fix2{typeof(StatsBase.moment),<:Integer}) = f.x
# the pth cumulant is a polynomial of degree p in the moments
PSIS._max_moment_required(f::Base.Fix2{typeof(StatsBase.cumulant),<:Integer}) = f.x
PSIS._max_moment_required(::Base.Fix2{typeof(StatsBase.percentile),<:Real}) = 0

end  # module
