# added here to avoid a dependency on LogExpFunctions
function logsumexp(x)
    xmax = maximum(x)
    return log(sum(xᵢ -> exp(xᵢ - xmax), x)) + xmax
end
