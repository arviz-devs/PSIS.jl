# added here to avoid a dependency on LogExpFunctions
function logsumexp(x)
    xmax = maximum(x)
    return log(sum(xᵢ -> exp(xᵢ - xmax), x)) + xmax
end

uniform_probabilities(T, npoints::Integer) = ((1:npoints) .- T(1//2)) ./ npoints

# p-th quartile of x, assuming x is sorted in increasing order
quartile(x, p) = x[fld(length(x) * p + 2, 4)] 
