struct SISResult{T,W<:AbstractArray{T},R} <: AbstractISResult
    log_weights::W
    r_eff::R
end

sis(logr, r_eff) = sis!(logr, r_eff)
sis!(logw, r_eff) = SISResult(logw, r_eff)
