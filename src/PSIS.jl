module PSIS

using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISPlots
export PSISResult
export psis, psis!, ess_is, importance_weights

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("weights.jl")
include("ess.jl")
include("recipes/plots.jl")

end
