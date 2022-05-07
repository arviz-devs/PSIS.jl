module PSIS

using LogExpFunctions: LogExpFunctions
using PrettyTables: PrettyTables
using Printf: @sprintf
using RecipesBase: RecipesBase
using Requires: Requires
using Statistics: Statistics

export PSISResult
export psis, psis!, ess_is
export ParetoShapePlot, paretoshapeplot, paretoshapeplot!

const PLOTTING_BACKEND = Ref(:Plots)

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("ess.jl")
include("recipes/definitions.jl")
include("recipes/plots.jl")

function __init__()
    Requires.@require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        using .Makie
        include("recipes/makie.jl")
    end
    Requires.@require Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f" begin
        using .Distributions: Distributions
        include("distributions.jl")
    end
end

end
