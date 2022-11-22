module PSIS

using LogExpFunctions: LogExpFunctions
using Printf: @sprintf
using Requires: Requires
using Statistics: Statistics

export PSISResult
export psis, psis!, ess_is

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")
include("ess.jl")
include("recipes/plots.jl")

function __init__()
    Requires.@require Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f" begin
        using .Distributions: Distributions
        include("distributions.jl")
    end
end

end
