struct TISResult{T,W<:AbstractArray{T},R} <: AbstractISResult
    log_weights::W
    r_eff::R
    log_weights_max::T
end

function tis(logr, r_eff)
    T = float(eltype(logr))
    logw = copyto!(similar(logr, T), logr)
    return tis!(logw, r_eff)
end
function tis!(logw::AbstractVecOrMat, r_eff)
    S = length(logw)
    logsumw = logsumexp(logw)
    logw_max = logsumw - log(oftype(logsumw, S)) / 2 # log(mean(r) * sqrt(S))
    logw .= min.(logw, logw_max)
    return TISResult(logw, r_eff, logw_max)
end
function tis!(logw::AbstractArray{T,3}, r_eff) where {T}
    nparams = size(logw, 1)
    logw_max = Vector{T}(undef, nparams)
    Threads.@threads for i in 1:nparams
        logwᵢ = @views logw[i, :, :]
        res = tis!(logwᵢ, r_eff[i])
        logw_max[i] = res.logw_max
    end
    return TISResult(logw, r_eff, logw_max)
end

function Base.show(io::IO, mime::MIME"text/plain", r::TISResult)
    invoke(Base.show, Tuple{IO,typeof(mime),AbstractISResult}, io, mime, r)
    print(io, "\n    log_weights_max: ", r.log_weights_max)
    return nothing
end
