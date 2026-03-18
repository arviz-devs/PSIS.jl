module PSIS

using DocStringExtensions: FIELDS
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISPlots
export PSISResult
export psis

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

end
