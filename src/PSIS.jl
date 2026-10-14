module PSIS

using DocStringExtensions: FIELDS
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISResult, ParetoDiagnostics
export pareto_k_table
export psis
export pareto_smooth
export pareto_khat,
    pareto_diagnose, pareto_min_ss, pareto_khat_threshold, pareto_convergence_rate

include("utils.jl")
include("generalized_pareto.jl")
include("diagnostics.jl")
include("tails.jl")
include("summary.jl")
include("smooth.jl")
include("core.jl")
include("ess.jl")

end
