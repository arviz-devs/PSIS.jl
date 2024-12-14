using Distributions, Plots, PSIS
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(PSIS, :DocTestSetup, :(using PSIS); recursive=true)

links = InterLinks(
    "MCMCDiagnosticTools" => "https://julia.arviz.org/MCMCDiagnosticTools/stable/"
)

makedocs(;
    modules=[PSIS],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo=Remotes.GitHub("arviz-devs", "PSIS.jl"),
    sitename="PSIS.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    pages=[
        "Home" => "index.md",
        "Plotting" => "plotting.md",
        "API" => "api.md",
        "Internal" => "internal.md",
    ],
    warnonly=:missing_docs,
    plugins=[links],
)

deploydocs(; repo="github.com/arviz-devs/PSIS.jl.git", devbranch="main")
