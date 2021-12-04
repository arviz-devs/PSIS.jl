module PSIS

using Distributions: Distributions
using LinearAlgebra: dot
using LogExpFunctions: logsumexp, softmax, softmax!
using Printf: @sprintf
using Requires: Requires
using Statistics: mean, median, quantile
using StatsBase: StatsBase

export PSISResult
export psis, psis!, paretoshapeplot, paretoshapeplot!

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")

function __init__()
    Requires.@require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        using .Makie
        include("recipes/makie.jl")
    end
end

end
