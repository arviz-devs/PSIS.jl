using CairoMakie, Distributions, Plots, PSIS
using Documenter

DocMeta.setdocmeta!(PSIS, :DocTestSetup, :(using PSIS); recursive=true)

makedocs(;
    modules=[PSIS],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/arviz-devs/PSIS.jl/blob/{commit}{path}#{line}",
    sitename="PSIS.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    pages=[
        "Home" => "index.md",
        "Plotting" => "plotting.md",
        "API" => "api.md",
        "Internal" => "internal.md",
    ],
)

deploydocs(; repo="github.com/arviz-devs/PSIS.jl.git", devbranch="main")
