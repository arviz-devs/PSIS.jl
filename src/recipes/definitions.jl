"""
    paretoshapeplot(values; showlines=false)
    paretoshapeplot!(values; showlines=false)

Plot the Pareto shape values of `values`.

`values` may be either an array of log importance ratios, in which case [`psis`](@ref) is
called, or a [`PSISResult`](@ref).

If `showlines=true`, then horizontal lines indicating the different Pareto shape thresholds
are drawn.

!!! note
    
    Plots.jl must be loaded to use these functions.
"""
paretoshapeplot, paretoshapeplot!
