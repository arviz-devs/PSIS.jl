const _PARETO_K_CATEGORIES = (:good, :bad, :very_bad, :failed, :missing)

struct ParetoKTable{
    T<:Real,E<:Union{Nothing,NamedTuple{_PARETO_K_CATEGORIES,<:NTuple{5,Real}}}
}
    threshold::T
    counts::NamedTuple{_PARETO_K_CATEGORIES,NTuple{5,Int}}
    min_ess::E
end

_category_name(category::Symbol) = replace(string(category), "_" => " ")

"""
    pareto_k_table(khat, threshold; ess=nothing) -> ParetoKTable
    pareto_k_table(result::PSISResult) -> ParetoKTable
    pareto_k_table(diagnostics::ParetoDiagnostics) -> ParetoKTable

Summarize Pareto k-hat values by counting them in each category:

  - `good`: `khat <= threshold`
  - `bad`: `threshold < khat <= 1`
  - `very bad`: `khat > 1`
  - `failed`: `khat` is `NaN`, i.e. the tail fit failed
  - `missing`: `khat` is `missing`

If `ess` is provided, it must have the same shape as `khat`, and the minimum
ESS in each category is also recorded.

The result displays as a summary table. If Tables.jl is loaded, it also
implements the Tables.jl interface, with one row per category. Packages whose
results contain Pareto k-hat values may overload this function to summarize them.
"""
function pareto_k_table(khat, threshold::Real; ess=nothing)
    k = as_array(khat)
    categories = map(Base.Fix2(_pareto_k_category, threshold), k)
    counts = map(c -> count(==(c), categories), _named_categories())
    min_ess = if isnothing(ess)
        nothing
    else
        e = as_array(ess)
        size(e) == size(k) ||
            throw(DimensionMismatch("ess must have the same shape as khat"))
        map(_named_categories()) do c
            values = [e[j] for j in eachindex(e, categories) if categories[j] === c]
            return isempty(values) ? float(eltype(e))(NaN) : minimum(values)
        end
    end
    return ParetoKTable(threshold, counts, min_ess)
end

_named_categories() = NamedTuple{_PARETO_K_CATEGORIES}(_PARETO_K_CATEGORIES)

function _pareto_k_category(k, threshold)
    ismissing(k) && return :missing
    isnan(k) && return :failed
    k <= threshold && return :good
    k <= 1 && return :bad
    return :very_bad
end

function _pareto_k_ranges(t::ParetoKTable)
    label = @sprintf("%.3g", t.threshold)
    return (
        good="(-Inf, $label]",
        bad="($label, 1]",
        very_bad="(1, Inf)",
        failed=missing,
        missing=missing,
    )
end

function Base.show(io::IO, ::MIME"text/plain", t::ParetoKTable)
    total = sum(t.counts)
    total == 0 && return print(io, "No parameter diagnostics.")
    ess = t.min_ess
    ranges = map(r -> coalesce(r, "——"), _pareto_k_ranges(t))
    rows = map(collect(_PARETO_K_CATEGORIES)) do c
        count = t.counts[c]
        perc = 100 * count / total
        ess_min = isnothing(ess) ? NaN : ess[c]
        return (
            category=c,
            range=ranges[c],
            desc="(" * _category_name(c) * ")",
            count_perc=(count, perc),
            ess_min=ess_min,
        )
    end
    rows = filter(r -> r.count_perc[1] > 0, rows)
    formats = (
        good=(),
        bad=(bold=true, color=:light_red),
        very_bad=(bold=true, color=:red),
        failed=(; color=:red),
        missing=(),
    )

    col_padding = " "
    col_delim = ""
    col_delim_tot = col_padding * col_delim * col_padding
    col_widths = [
        maximum(r -> length(r.range), rows),
        maximum(r -> length(r.desc), rows),
        maximum(r -> ndigits(r.count_perc[1]), rows),
        floor(Int, log10(maximum(r -> r.count_perc[2], rows))) + 6,
    ]

    println(io, "Pareto k-hat diagnostic summary:")
    printstyled(
        io,
        col_padding,
        " "^col_widths[1],
        col_delim_tot,
        " "^col_widths[2],
        col_delim_tot,
        isnothing(ess) ? "Count" : _pad_right("Count", col_widths[3] + col_widths[4] + 1);
        bold=true,
    )
    if !isnothing(ess)
        printstyled(io, col_delim_tot, "Min. ESS"; bold=true)
    end
    for r in rows
        count, perc = r.count_perc
        perc_str = "($(round(perc; digits=1))%)"
        println(io)
        print(io, col_padding, _pad_left(r.range, col_widths[1]), col_delim_tot)
        print(io, _pad_right(r.desc, col_widths[2]), col_delim_tot)
        format = formats[r.category]
        printstyled(io, _pad_left(count, col_widths[3]); format...)
        printstyled(io, " ", _pad_right(perc_str, col_widths[4]); format...)
        if !isnothing(ess)
            ess_min_string = if isfinite(r.ess_min) && r.category === :good
                string(floor(Int, r.ess_min))
            else
                "——"
            end
            print(io, col_delim_tot, ess_min_string)
        end
    end
    return nothing
end

_pad_left(s, nchars) = " "^(nchars - length("$s")) * "$s"
_pad_right(s, nchars) = "$s" * " "^(nchars - length("$s"))
