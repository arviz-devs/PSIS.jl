var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"PSISResult\npsis\nparetoshapeplot\ness_is","category":"page"},{"location":"api/#PSIS.PSISResult","page":"API","title":"PSIS.PSISResult","text":"PSISResult\n\nResult of Pareto-smoothed importance sampling (PSIS) using psis.\n\nProperties\n\nlog_weights: un-normalized Pareto-smoothed log weights\nweights: normalized Pareto-smoothed weights (allocates a copy)\npareto_shape: Pareto k=ξ shape parameter\nnparams: number of parameters in log_weights\nndraws: number of draws in log_weights\nnchains: number of chains in log_weights\nreff: the ratio of the effective sample size of the unsmoothed importance ratios and the actual sample size.\ness: estimated effective sample size of estimate of mean using smoothed importance samples (see ess_is)\nlog_weights_norm: the logarithm of the normalization constant of log_weights\ntail_length: length of the upper tail of log_weights that was smoothed\ntail_dist: the generalized Pareto distribution that was fit to the tail of log_weights\n\nDiagnostic\n\nThe pareto_shape parameter k=ξ of the generalized Pareto distribution tail_dist can be used to diagnose reliability and convergence of estimates using the importance weights [VehtariSimpson2021].\n\nif k  frac13, importance sampling is stable, and importance sampling (IS) and PSIS both are reliable.\nif k  frac12, then the importance ratio distributon has finite variance, and the central limit theorem holds. As k approaches the upper bound, IS becomes less reliable, while PSIS still works well but with a higher RMSE.\nif frac12  k  07, then the variance is infinite, and IS can behave quite poorly. However, PSIS works well in this regime.\nif 07  k  1, then it quickly becomes impractical to collect enough importance weights to reliably compute estimates, and importance sampling is not recommended.\nif k  1, then neither the variance nor the mean of the raw importance ratios exists. The convergence rate is close to zero, and bias can be large with practical sample sizes.\n\nSee paretoshapeplot for a diagnostic plot.\n\n[VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021). Pareto smoothed importance sampling. arXiv:1507.02646v7 [stat.CO]\n\n\n\n\n\n","category":"type"},{"location":"api/#PSIS.psis","page":"API","title":"PSIS.psis","text":"psis(log_ratios, reff = 1.0; kwargs...) -> PSISResult\npsis!(log_ratios, reff = 1.0; kwargs...) -> PSISResult\n\nCompute Pareto smoothed importance sampling (PSIS) log weights [VehtariSimpson2021].\n\nWhile psis computes smoothed log weights out-of-place, psis! smooths them in-place.\n\nArguments\n\nlog_ratios: an array of logarithms of importance ratios, with one of the following sizes:\n(ndraws,): a vector of draws for a single parameter from a single chain\n(nparams, ndraws): a matrix of draws for a multiple parameter from a single chain\n(nparams, ndraws, nchains): an array of draws for multiple parameters from multiple chains, e.g. as might be generated with Markov chain Monte Carlo.\nreff::Union{Real,AbstractVector}: the ratio(s) of effective sample size of log_ratios and the actual sample size reff = ess/(ndraws * nchains), used to account for autocorrelation, e.g. due to Markov chain Monte Carlo.\n\nKeywords\n\nimproved=false: If true, use the adaptive empirical prior of [Zhang2010]. If false, use the simpler prior of [ZhangStephens2009], which is also used in [VehtariSimpson2021].\nwarn=true: If true, warning messages are delivered\n\nReturns\n\nresult: a PSISResult object containing the results of the Pareto-smoothing.\n\nA warning is raised if the Pareto shape parameter k  07. See PSISResult for details and paretoshapeplot for a diagnostic plot.\n\n[VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021). Pareto smoothed importance sampling. arXiv:1507.02646v7 [stat.CO]\n\n[ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009) A New and Efficient Estimation Method for the Generalized Pareto Distribution, Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017\n\n[Zhang2010]: Jin Zhang (2010) Improving on Estimation for the Generalized Pareto Distribution, Technometrics, 52:3, 335-339, DOI: 10.1198/TECH.2010.09206\n\n\n\n\n\n","category":"function"},{"location":"api/#PSIS.paretoshapeplot","page":"API","title":"PSIS.paretoshapeplot","text":"paretoshapeplot(values; kwargs...)\nparetoshapeplot!(values; kwargs...)\n\nPlot shape parameters of fitted Pareto tail distributions for diagnosing convergence.\n\nArguments\n\nvalues: may be either a vector of Pareto shape parameters or a PSISResult.\n\nKeywords\n\nshowlines=false: if true, plot horizontal lines indicating relevant Pareto shape thresholds are drawn. See PSISResult for explanation of thresholds.\nbackend::Symbol: backend to use for plotting, defaulting to :Plots, unless :Makie is available.\n\nAll remaining keywords are passed to the plotting backend.\n\nSee psis, PSISResult.\n\nnote: Note\nPlots.jl or a Makie.jl backend must be loaded to use these functions.\n\nExamples\n\nusing PSIS, Distributions, Plots\nproposal = Normal()\ntarget = TDist(7)\nx = rand(proposal, 100, 1_000)\nlog_ratios = logpdf.(target, x) .- logpdf.(proposal, x)\nresult = psis(log_ratios)\n\nPlot with Plots.jl.\n\nusing Plots\nplot(result; showlines=true)\n\nPlot with GLMakie.jl.\n\nusing GLMakie\nplot(result; showlines=true)\n\n\n\n\n\n","category":"function"},{"location":"api/#PSIS.ess_is","page":"API","title":"PSIS.ess_is","text":"ess_is(weights; reff=1)\n\nEstimate effective sample size (ESS) for importance sampling over the sample dimensions.\n\nGiven normalized weights w_1n, the ESS is estimated using the L2-norm of the weights:\n\nmathrmESS(w_1n) = fracr_mathrmeffsum_i=1^n w_i^2\n\nwhere r_mathrmeff is the relative efficiency of the log_weights.\n\ness_is(result::PSISResult; bad_shape_missing=true)\n\nEstimate ESS for Pareto-smoothed importance sampling.\n\nnote: Note\nESS estimates for Pareto shape values k  07, which are unreliable and misleadingly high, are set to missing. To avoid this, set bad_shape_missing=false.\n\n\n\n\n\n","category":"function"},{"location":"internal/#Internal","page":"Internal","title":"Internal","text":"","category":"section"},{"location":"internal/","page":"Internal","title":"Internal","text":"Modules = [PSIS]\nPublic = false","category":"page"},{"location":"internal/#PSIS.GeneralizedParetoKnownMu","page":"Internal","title":"PSIS.GeneralizedParetoKnownMu","text":"GeneralizedParetoKnownMu(μ)\n\nRepresents a GeneralizedPareto where mu is known.\n\n\n\n\n\n","category":"type"},{"location":"internal/#PSIS.GeneralizedParetoKnownMuTheta","page":"Internal","title":"PSIS.GeneralizedParetoKnownMuTheta","text":"GeneralizedParetoKnownMuTheta(μ, θ)\n\nRepresents a GeneralizedPareto where mu and theta=fracxisigma are known.\n\n\n\n\n\n","category":"type"},{"location":"internal/#PSIS.plotting_backend!-Tuple{Symbol}","page":"Internal","title":"PSIS.plotting_backend!","text":"plotting_backend!(backend::Symbol)\n\nSet default plotting backend. Valid values are :Plots and :Makie.\n\n\n\n\n\n","category":"method"},{"location":"internal/#StatsBase.fit-Tuple{PSIS.GeneralizedParetoKnownMu, AbstractArray}","page":"Internal","title":"StatsBase.fit","text":"fit(g::GeneralizedParetoKnownMu, x; kwargs...)\n\nFit a GeneralizedPareto with known location μ to the data x.\n\nThe fit is performed using the Empirical Bayes method of [ZhangStephens2009][Zhang2010].\n\nKeywords\n\nsorted::Bool=issorted(x): If true, x is assumed to be sorted. If false, a sorted copy of x is made.\nimproved::Bool=true: If true, use the adaptive empirical prior of [Zhang2010]. If false, use the simpler prior of [ZhangStephens2009].\nmin_points::Int=30: The minimum number of quadrature points to use when estimating the posterior mean of theta = fracxisigma.\n\n[ZhangStephens2009]: Jin Zhang & Michael A. Stephens (2009) A New and Efficient Estimation Method for the Generalized Pareto Distribution, Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017\n\n[Zhang2010]: Jin Zhang (2010) Improving on Estimation for the Generalized Pareto Distribution, Technometrics, 52:3, 335-339, DOI: 10.1198/TECH.2010.09206\n\n\n\n\n\n","category":"method"},{"location":"plotting/#Plotting-PSIS-results","page":"Plotting","title":"Plotting PSIS results","text":"","category":"section"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"PSIS.jl includes plotting recipes for PSISResult using any Plots.jl or Makie.jl backend, as well as the utility plotting function paretoshapeplot.","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"We demonstrate this with a simple example.","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"using Random # hide\nRandom.seed!(42) # hide\nusing PSIS, Distributions\nproposal = Normal()\ntarget = TDist(7)\nx = rand(proposal, 20, 1_000)\nlog_ratios = logpdf.(target, x) .- logpdf.(proposal, x)\nresult = psis(log_ratios)","category":"page"},{"location":"plotting/#Plots.jl","page":"Plotting","title":"Plots.jl","text":"","category":"section"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"PSISResult objects can be plotted directly:","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"using Plots\nPlots.plot(result; showlines=true, marker=:+, legend=false, linewidth=2)","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"This is equivalent to calling paretoshapeplot(result; kwargs...).","category":"page"},{"location":"plotting/#Makie.jl","page":"Plotting","title":"Makie.jl","text":"","category":"section"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"The same syntax is supported with Makie.jl backends.","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"using CairoMakie\nCairoMakie.activate!(; type = \"svg\") # hide\nMakie.inline!(true) # hide\nMakie.plot(result; showlines=true, marker=:+)","category":"page"},{"location":"plotting/#Selecting-the-backend","page":"Plotting","title":"Selecting the backend","text":"","category":"section"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"If a Makie backend is loaded, then by default paretoshapeplot and paretoshapeplot! will use that backend; otherwise, a Plots.jl backend is used if available. However, both functions accept a backend keyword that can be used to specify the backend if for some reason both are loaded.","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"paretoshapeplot(result; backend=:Plots)","category":"page"},{"location":"plotting/","page":"Plotting","title":"Plotting","text":"paretoshapeplot(result; backend=:Makie)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = PSIS","category":"page"},{"location":"#PSIS","page":"Home","title":"PSIS","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Coverage) (Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages) (Image: Powered by NumFOCUS)","category":"page"},{"location":"","page":"Home","title":"Home","text":"PSIS.jl implements the Pareto smoothed importance sampling (PSIS) algorithm from [VehtariSimpson2021].","category":"page"},{"location":"","page":"Home","title":"Home","text":"Given a set of importance weights used in some estimator, PSIS both improves the reliability of the estimates by smoothing the importance weights and acts as a diagnostic of the reliability of the estimates.","category":"page"},{"location":"","page":"Home","title":"Home","text":"See psis for details.","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In this example, we use PSIS to smooth log importance ratios for importance sampling 30 isotropic Student t-distributed parameters using standard normal distributions as proposals.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Random # hide\nRandom.seed!(42) # hide\nusing PSIS, Distributions\nproposal = Normal()\ntarget = TDist(7)\nx = rand(proposal, 30, 1_000)\nlog_ratios = logpdf.(target, x) .- logpdf.(proposal, x)\nresult = psis(log_ratios)\nnothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"result # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"As indicated by the warnings, this is a poor choice of a proposal distribution, and estimates are unlikely to converge (see PSISResult for an explanation of the shape thresholds).","category":"page"},{"location":"","page":"Home","title":"Home","text":"When running PSIS with many parameters, it is useful to plot the Pareto shape values to diagnose convergence. See Plotting PSIS results for examples.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[VehtariSimpson2021]: Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J. (2021). Pareto smoothed importance sampling. arXiv:1507.02646v7 [stat.CO]","category":"page"}]
}
