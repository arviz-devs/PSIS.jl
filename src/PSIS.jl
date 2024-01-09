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
include("tails.jl")
include("expectand.jl")
include("diagnostics.jl")
include("pareto_diagnose.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

@static if !isdefined(Base, :get_extension)
    using Requires: @require
    function __init__()
        @require StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" begin
            include("../ext/PSISStatsBaseExt.jl")
        end
    end
end

end
