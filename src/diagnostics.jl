"""
    ParetoDiagnostics

Diagnostic information for Pareto-smoothed importance sampling.[^VehtariSimpson2021]

$FIELDS

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]
"""
struct ParetoDiagnostics{TK,TKM,TS,TR}
    "The estimated Pareto shape ``\\hat{k}`` for each parameter."
    pareto_shape::TK
    "The sample-size-dependent Pareto shape threshold ``k_\\mathrm{threshold}`` needed for a
    reliable Pareto-smoothed estimate (i.e. to have small probability of large error)."
    pareto_shape_threshold::TKM
    "The estimated minimum sample size needed for a reliable Pareto-smoothed estimate (i.e.
    to have small probability of large error)."
    min_sample_size::TS
    "The estimated relative convergence rate of the RMSE of the Pareto-smoothed estimate."
    convergence_rate::TR
end

pareto_shape_threshold(sample_size::Real) = 1 - inv(log10(sample_size))

function min_sample_size(pareto_shape::Real)
    min_ss = exp10(inv(1 - max(0, pareto_shape)))
    return pareto_shape > 1 ? oftype(min_ss, Inf) : min_ss
end
min_sample_size(pareto_shape::AbstractArray) = map(min_sample_size, pareto_shape)

function convergence_rate(k::AbstractArray{<:Real}, S::Real)
    return convergence_rate.(k, S)
end
function convergence_rate(k::Real, S::Real)
    T = typeof((one(S) * 1^zero(k) * oneunit(k)) / (one(S) * 1^zero(k)))
    k < 0 && return oneunit(T)
    k > 1 && return zero(T)
    k == 1//2 && return T(1 - inv(log(S)))
    return T(
        max(
            0,
            (2 * (k - 1) * S^(2k) - (2k - 1) * S^(2k - 1) + S) /
            ((S - 1) * (1 - S^(2k - 1))),
        ),
    )
end

"""
    check_pareto_diagnostics(diagnostics::ParetoDiagnostics)

Check the diagnostics in [`ParetoDiagnostics`](@ref) and issue warnings if necessary.
"""
function check_pareto_diagnostics(diag::ParetoDiagnostics)
    categories = _diagnostic_intervals(diag)
    category_assignments = _diagnostic_category_assignments(diag)
    nparams = length(diag.pareto_shape)
    for (category, inds) in pairs(category_assignments)
        count = length(inds)
        count > 0 || continue
        perc = round(Int, 100 * count / nparams)
        msg = if category === :failed
            "The generalized Pareto distribution could not be fit to the tail draws. " *
            "Total number of draws should in general exceed 25, and the tail draws must " *
            "be finite."
        elseif category === :very_bad
            "All estimates are unreliable. If the distribution of draws is bounded, " *
            "further draws may improve the estimates, but it is not possible to predict " *
            "whether any feasible sample size is sufficient."
        elseif category === :bad
            ss_max = ceil(maximum(i -> diag.min_sample_size[i], inds))
            "Sample size is too small and must be larger than " *
            "$(@sprintf("%.10g", ss_max)) for all estimates to be reliable."
        elseif category === :high_bias
            "Bias dominates RMSE, and variance-based MCSE estimates are underestimated."
        else
            continue
        end
        suffix =
            category === :failed ? "" : " (k ∈ $(_interval_string(categories[category])))"
        prefix = if nparams > 1
            msg = lowercasefirst(msg)
            prefix = "For $count parameters ($perc%), "
        else
            ""
        end
        @warn "$prefix$msg$suffix"
    end
end

function _compute_diagnostics(pareto_shape, sample_size)
    return ParetoDiagnostics(
        pareto_shape,
        pareto_shape_threshold(sample_size),
        min_sample_size(pareto_shape),
        convergence_rate(pareto_shape, sample_size),
    )
end

function _interval_string(i::IntervalSets.Interval)
    l = IntervalSets.isleftopen(i) || !isfinite(minimum(i)) ? "(" : "["
    r = IntervalSets.isrightopen(i) || !isfinite(maximum(i)) ? ")" : "]"
    imin, imax = IntervalSets.endpoints(i)
    return "$l$(@sprintf("%.1g", imin)), $(@sprintf("%.1g", imax))$r"
end

function _diagnostic_intervals(diag::ParetoDiagnostics)
    khat_thresh = diag.pareto_shape_threshold
    return (
        good=IntervalSets.ClosedInterval(-Inf, khat_thresh),
        bad=IntervalSets.Interval{:open,:closed}(khat_thresh, 1),
        very_bad=IntervalSets.Interval{:open,:closed}(1, Inf),
        high_bias=IntervalSets.Interval{:open,:closed}(0.7, 1),
    )
end

function _diagnostic_category_assignments(diagnostics)
    intervals = _diagnostic_intervals(diagnostics)
    result_counts = map(intervals) do interval
        return findall(∈(interval), diagnostics.pareto_shape)
    end
    failed = findall(isnan, diagnostics.pareto_shape)
    return merge(result_counts, (; failed))
end

function Base.show(io::IO, ::MIME"text/plain", diag::ParetoDiagnostics)
    nparams = length(diag.pareto_shape)
    println(io, "ParetoDiagnostics with $nparams parameters")
    return _print_pareto_diagnostics_summary(io, diag; newline_at_end=false)
end

function _print_pareto_diagnostics_summary(io::IO, diag::ParetoDiagnostics; kwargs...)
    k = as_array(diag.pareto_shape)
    category_assignments = NamedTuple{(:good, :bad, :very_bad, :failed)}(
        _diagnostic_category_assignments(diag)
    )
    category_intervals = _diagnostic_intervals(diag)
    npoints = length(k)
    rows = map(collect(pairs(category_assignments))) do (desc, inds)
        interval = desc === :failed ? "--" : _interval_string(category_intervals[desc])
        return (; interval, desc, count=length(inds))
    end
    return _print_pareto_diagnostics_summary(io::IO, rows, npoints; kwargs...)
end

function _print_pareto_diagnostics_summary(io::IO, _rows, npoints; kwargs...)
    rows = filter(r -> r.count > 0, _rows)
    header = ["", "", "Count"]
    alignment = [:r, :l, :l]
    if length(first(rows)) > 3
        push!(header, "Min. ESS")
        push!(alignment, :r)
    end
    formatters = (
        (v, i, j) -> j == 2 ? replace(string(v), '_' => " ") : v,
        (v, i, j) -> j == 3 ? "$v ($(round(v * (100 // npoints); digits=1))%)" : v,
        (v, i, j) -> j == 4 ? (rows[i].desc === :good ? "$(floor(Int, v))" : "——") : v,
    )
    highlighters = (
        PrettyTables.Highlighter(
            (data, i, j) -> (j == 3 && data[i][2] === :bad);
            bold=true,
            foreground=:light_red,
        ),
        PrettyTables.Highlighter(
            (data, i, j) -> (j == 3 && data[i][2] === :very_bad); bold=true, foreground=:red
        ),
        PrettyTables.Highlighter(
            (data, i, j) -> (j == 3 && data[i][2] === :failed); foreground=:red
        ),
    )

    PrettyTables.pretty_table(
        io,
        rows;
        header,
        alignment,
        alignment_anchor_regex=Dict(3 => [r"\s"]),
        hlines=:none,
        vlines=:none,
        formatters,
        highlighters,
        kwargs...,
    )
    return nothing
end

_pad_left(s, nchars) = " "^max(nchars - length("$s"), 0) * "$s"
_pad_right(s, nchars) = "$s" * " "^max(0, nchars - length("$s"))
