module PSISTablesExt

using PSIS: PSIS, ParetoKTable
using Tables: Tables

Tables.istable(::Type{<:ParetoKTable}) = true
Tables.columnaccess(::Type{<:ParetoKTable}) = true
function Tables.columns(t::ParetoKTable)
    total = sum(t.counts)
    cols = (
        category=collect(map(PSIS._category_name, PSIS._PARETO_K_CATEGORIES)),
        range=collect(Union{Missing,String}, values(PSIS._pareto_k_ranges(t))),
        count=collect(values(t.counts)),
        proportion=collect(values(t.counts) ./ total),
    )
    isnothing(t.min_ess) && return cols
    return merge(cols, (; min_ess=collect(values(t.min_ess))))
end

end
