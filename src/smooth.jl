function pareto_smooth(
    x::AbstractArray;
    r_eff::Union{Real,AbstractArray}=1,
    is_log_scale::Bool=false,
    tail::Symbol=(is_log_scale ? :right : :both),
)
    x_smoothed = similar(x)
    copyto!(x_smoothed, x)
    _, diagnostics = pareto_diagnose!(x_smoothed; smooth=true, r_eff, is_log_scale, tail)
    return x_smoothed, diagnostics
end
