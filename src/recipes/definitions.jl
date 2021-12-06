"""
    paretoshapeplot(values; showlines=false)
    paretoshapeplot!(values; showlines=false)

Plot shape parameters of fitted Pareto tail distributions for diagnosing convergence.

`values` may be either a vector of Pareto shape parameters or a [`PSISResult`](@ref).

If `showlines=true`, then horizontal lines indicating the different Pareto shape thresholds
are drawn.

!!! note
    
    Plots.jl must be loaded to use these functions.
"""
paretoshapeplot, paretoshapeplot!
