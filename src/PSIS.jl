module PSIS

using Compat: @constprop
using DocStringExtensions: FIELDS
using IntervalSets: IntervalSets
using LogExpFunctions: LogExpFunctions
using PrettyTables: PrettyTables
using Printf: @sprintf
using Statistics: Statistics

const EXTENSIONS_SUPPORTED = isdefined(Base, :get_extension)

export PSISPlots
export ParetoDiagnostics, PSISResult
export pareto_diagnose, pareto_smooth, psis, psis!
export check_pareto_diagnostics, ess_is

include("utils.jl")
include("generalized_pareto.jl")
include("tails.jl")
include("expectand.jl")
include("diagnostics.jl")
include("pareto_diagnose.jl")
include("pareto_smooth.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

if !EXTENSIONS_SUPPORTED
    using Requires: @require
end

function __init__()
    @static if !EXTENSIONS_SUPPORTED
        @require StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" begin
            include("../ext/PSISStatsBaseExt.jl")
        end
    end
end

end
