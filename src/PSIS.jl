module PSIS

using DocStringExtensions: FIELDS
using IntervalSets: IntervalSets
using LogExpFunctions: LogExpFunctions
using PrettyTables: PrettyTables
using Printf: @sprintf
using Statistics: Statistics

export PSISPlots
export PSISResult
export psis, psis!, ess_is
export pareto_diagnose

include("utils.jl")
include("generalized_pareto.jl")
include("diagnostics.jl")
include("pareto_diagnose.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

end
