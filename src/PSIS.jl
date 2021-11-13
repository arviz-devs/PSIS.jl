module PSIS

using Distributions: Distributions
using LogExpFunctions: logsumexp
using Statistics: mean, quantile
using StatsBase: StatsBase
using LinearAlgebra: dot
using Printf: @sprintf

export PSISResult, psis, psis!

abstract type AbstractISResult end

function Base.getproperty(r::AbstractISResult, k::Symbol)
    if k === :weights
        log_weights = getfield(r, :log_weights)
        log_sum_weights = if ndims(log_weights) == 3
            logsumexp(log_weights; dims=(2, 3))
        else
            logsumexp(log_weights)
        end
        # for arrays with named axes and dimensions, this preserves the names
        # in the result
        weights = similar(log_weights)
        weights .= exp.(log_weights .- log_sum_weights)
        return weights
    end
    if k === :nparams
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 1 ? 1 : size(log_weights, 1)
    end
    if k === :ndraws
        log_weights = getfield(r, :log_weights)
        return ndims(log_weights) == 3 ? prod(size(log_weights)[2:3]) : length(log_weights)
    end
    if k === :nchains
        log_weights = getfield(r, :log_weights)
        d = ndims(log_weights)
        return d == 1 ? 1 : size(log_weights, d)
    end
    k === :pareto_k && return pareto_k(r)
    return getfield(r, k)
end

function Base.propertynames(r::AbstractISResult)
    return [fieldnames(typeof(r))..., :weights, :nparams, :ndraws, :nchains]
end

function Base.show(io::IO, ::MIME"text/plain", r::AbstractISResult)
    println(io, typeof(r), ":")
    println(io, "    (nparams, ndraws, nchains): ", (r.nparams, r.ndraws, r.nchains))
    print(io, "    r_eff: ", r.r_eff)
    return nothing
end

pareto_k(::AbstractISResult) = missing

include("utils.jl")
include("generalized_pareto.jl")
include("standard.jl")
include("truncated.jl")
include("smoothed.jl")

end
