module PSIS

using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISPlots
export PSISResult
export psis, psis!, ess_is

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

end
