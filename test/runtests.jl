using PSIS
using Test

@testset "PSIS.jl" begin
    include("generalized_pareto.jl")
    include("core.jl")
    include("ess.jl")
    include("plotting_backend.jl")
    include("plots.jl")
    include("makie.jl")
end
