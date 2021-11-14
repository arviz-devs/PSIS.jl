uniform_probabilities(T, npoints::Integer) = ((1:npoints) .- T(1//2)) ./ npoints
