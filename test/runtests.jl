using PSIS
using Test

@testset "PSIS.jl" begin
    include("utils.jl")
    include("generalized_pareto.jl")
    include("core.jl")
    include("weights.jl")
    include("ess.jl")
    include("plots.jl")
end
