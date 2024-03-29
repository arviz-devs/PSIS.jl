# PSIS

[![Docs](https://img.shields.io/badge/docs-ArviZ-blue.svg)](https://julia.arviz.org/PSIS)
[![Build Status](https://github.com/arviz-devs/PSIS.jl/workflows/CI/badge.svg)](https://github.com/arviz-devs/PSIS.jl/actions)
[![Coverage](https://codecov.io/gh/arviz-devs/PSIS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/arviz-devs/PSIS.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

PSIS.jl implements the Pareto smoothed importance sampling (PSIS) algorithm from Vehtari et al, 2021.
Given a set of importance weights used in some estimator, PSIS both improves the reliability of the estimates by smoothing the importance weights and acts as a diagnostic of the reliability of the estimates.

Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021). Pareto smoothed importance sampling.
[arXiv:1507.02646v7](https://arxiv.org/abs/1507.02646v7) [stat.CO]

# Related Packages

The following packages also implement PSIS:

- [PSIS](https://github.com/avehtari/PSIS): Matlab and Python reference implementations
- [loo](https://github.com/stan-dev/loo)
- [arviz](https://github.com/arviz-devs/arviz)
- [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl)
