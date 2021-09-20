using PSIS
using Test

@testset "PSIS.jl" begin
    include("generalized_pareto.jl")
    include("psis.jl")
end
