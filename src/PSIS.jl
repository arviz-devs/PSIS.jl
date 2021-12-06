module PSIS

using Distributions: Distributions
using LinearAlgebra: dot
using LogExpFunctions: logsumexp, softmax, softmax!
using Printf: @sprintf
using RecipesBase: RecipesBase
using Statistics: mean, median, quantile
using StatsBase: StatsBase

export PSISResult
export psis, psis!, paretoshapeplot, paretoshapeplot!

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")

include("recipes/definitions.jl")
include("recipes/plots.jl")

end
