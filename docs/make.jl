using Distributions, Plots, PSIS
using Documenter
using DocumenterCitations
using DocumenterInterLinks

DocMeta.setdocmeta!(PSIS, :DocTestSetup, :(using PSIS); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:numeric)

links = InterLinks(
    "MCMCDiagnosticTools" => "https://julia.arviz.org/MCMCDiagnosticTools/stable/"
)

makedocs(;
    modules=[PSIS],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo=Remotes.GitHub("arviz-devs", "PSIS.jl"),
    sitename="PSIS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=[joinpath("assets", "citations.css")],
    ),
    pages=[
        "Home" => "index.md",
        "Plotting" => "plotting.md",
        "API" => "api.md",
        "Internal" => "internal.md",
        "References" => "references.md",
    ],
    doctestfilters=[r"└.*"],  # ignore locations in warning messages
    warnonly=:missing_docs,
    plugins=[bib, links],
)

deploydocs(; repo="github.com/arviz-devs/PSIS.jl.git", devbranch="main")
