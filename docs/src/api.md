# API

## Core functionality

```@docs
PSISResult
psis
```

## Tail diagnostics and smoothing

```@docs
ParetoDiagnostics
pareto_khat
pareto_diagnose
pareto_smooth
pareto_min_ss
pareto_khat_threshold
pareto_convergence_rate
```

## Diagnostics from an existing fit

These scalar helpers do not fit observations. Broadcast them to work with arrays
of already estimated k-hat values.

```@docs
ps_min_ss
ps_khat_threshold
ps_convergence_rate
```
