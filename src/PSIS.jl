module PSIS

using DocStringExtensions: FIELDS
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Statistics: Statistics

export PSISResult
export psis
export pareto_smooth
export ps_min_ss, ps_khat_threshold, ps_convergence_rate
export pareto_khat,
    pareto_diagnose, pareto_min_ss, pareto_khat_threshold, pareto_convergence_rate

include("utils.jl")
include("generalized_pareto.jl")
include("diagnostics.jl")
include("tails.jl")
include("diagnose.jl")
include("smooth.jl")
include("core.jl")
include("ess.jl")

end
