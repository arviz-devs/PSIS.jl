```@meta
CurrentModule = PSIS
```

# PSIS

PSIS.jl implements the Pareto smoothed importance sampling (PSIS) algorithm from [^VehtariSimpson2021].

Given a set of importance weights used in some estimator, PSIS both improves the reliability of the estimates by smoothing the importance weights and acts as a diagnostic of the reliability of the estimates.

See [`psis`](@ref) for details.

[^VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021).
    Pareto smoothed importance sampling.
    [arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]

