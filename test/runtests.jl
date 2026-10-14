using PSIS
using Test

@testset "PSIS.jl" begin
    include("utils.jl")
    include("generalized_pareto.jl")
    include("diagnostics.jl")
    include("tails.jl")
    include("pareto_k_table.jl")
    include("smooth.jl")
    include("psis_diagnostics.jl")
    include("importance_sampling.jl")
    include("ess.jl")
end
