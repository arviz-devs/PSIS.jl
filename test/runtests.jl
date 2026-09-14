using PSIS
using Test

@testset "PSIS.jl" begin
    include("utils.jl")
    include("generalized_pareto.jl")
    include("diagnostics.jl")
    include("tails.jl")
    include("diagnose.jl")
    include("core.jl")
    include("ess.jl")
end
