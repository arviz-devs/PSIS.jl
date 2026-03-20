module PSIS

using DocStringExtensions: FIELDS
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISResult
export pareto_diagnose, pareto_smooth, psis

include("utils.jl")
include("generalized_pareto.jl")
include("diagnose.jl")
include("smooth.jl")
include("core.jl")
include("ess.jl")

end
