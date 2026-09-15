function Base.show(io::IO, ::MIME"text/plain", diagnostics::ParetoDiagnostics)
    nparams = length(as_array(diagnostics.khat))
    parameter_label = nparams == 1 ? "parameter" : "parameters"
    println(io, "ParetoDiagnostics with $nparams $parameter_label")
    threshold = min(0.7, diagnostics.khat_threshold)
    return _print_pareto_khat_summary(io, diagnostics.khat, threshold)
end

function _print_pareto_khat_summary(io::IO, khat, threshold; ess_is=nothing)
    k = as_array(khat)
    isempty(k) && return print(io, "No parameter diagnostics.")
    ess = isnothing(ess_is) ? nothing : as_array(ess_is)
    label = @sprintf("%.3g", threshold)
    categories = (
        ("(-Inf, $label]", "good", <=(threshold)),
        ("($label, 1]", "bad", k -> threshold < k <= 1),
        ("(1, Inf)", "very bad", >(1)),
        ("——", "failed", isnan),
    )
    rows = map(categories) do (range, desc, cond)
        inds = findall(cond, k)
        count = length(inds)
        perc = 100 * count / length(k)
        ess_min = if isnothing(ess) || count == 0 || desc == "failed"
            NaN
        else
            minimum(view(ess, inds))
        end
        return (range=range, desc=desc, count_perc=(count, perc), ess_min=ess_min)
    end
    rows = filter(r -> r.count_perc[1] > 0, rows)
    formats = Dict(
        "good" => (),
        "bad" => (bold=true, color=:light_red),
        "very bad" => (bold=true, color=:red),
        "failed" => (; color=:red),
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
        format = formats[r.desc]
        printstyled(io, _pad_left(count, col_widths[3]); format...)
        printstyled(io, " ", _pad_right(perc_str, col_widths[4]); format...)
        if !isnothing(ess)
            ess_min_string = if isfinite(r.ess_min) && r.desc == "good"
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
