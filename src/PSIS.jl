module PSIS

using Distributions: Distributions
using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using RecipesBase: RecipesBase
using Requires: Requires
using Statistics: Statistics
using StatsBase: StatsBase

export PSISResult
export psis, psis!
export ParetoShapePlot, paretoshapeplot, paretoshapeplot!

const PLOTTING_BACKEND = Ref(:Plots)

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("recipes/definitions.jl")
include("recipes/plots.jl")

function __init__()
    Requires.@require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        using .Makie
        include("recipes/makie.jl")
    end
end

end
