module PSIS

using Distributions: Distributions
using LinearAlgebra: dot
using LogExpFunctions: logsumexp, softmax, softmax!
using Printf: @sprintf
using Statistics: mean, median, quantile
using StatsBase: StatsBase

export PSISResult
export psis, psis!

include("utils.jl")
include("generalized_pareto.jl")
include("core.jl")

end
