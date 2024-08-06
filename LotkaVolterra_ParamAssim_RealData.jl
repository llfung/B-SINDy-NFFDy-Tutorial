## Data assimilation to estimate parameters of the Lotka-Volterra model using Turing.jl
# Adapted from the Turing.jl documentation: https://turinglang.org/docs/tutorials/10-bayesian-differential-equations/

## Initialisation
using Turing, DifferentialEquations, LinearAlgebra, Distributions, StatsBase
import MAT

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots
plotlyjs()

## Forward problem - Lotka-Volterra model
# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    du[1] = (α - β * y) * x # prey
    du[2] = (δ * x - γ) * y # predator

    return nothing
end

## Import data
matfile = MAT.matread("./data/LynxHare.mat")
data = matfile["LynxHare"]
data = collect(data[:,[2,1]]')

# Define initial-value problem.
u0 = data[:,1] # Initial state.
p = [0.533 0.0263 0.98 0.0276] # Values from literature, not affecting the learning as p here is just a placeholder.
tspan = (0.0, 20.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)
# sol = solve(prob, Tsit5(); p=p, saveat=1.0)

## Inverse problem - Bayesian inference
@model function fitlv(data, prob)
    # Prior distributions.
    σ2 ~ InverseGamma(2, 3)
    # σ2 = 2.7^2
    α ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)
    β ~ truncated(Normal(0.02, 0.01); lower=0, upper=0.1)
    γ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)
    δ ~ truncated(Normal(0.02, 0.01); lower=0, upper=0.1)

    # Simulate Lotka-Volterra model. 
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=1.0)

    # Observations.
    for i in 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ2 * I)
    end

    return nothing
end

model = fitlv(data, prob)

# Sample with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(), 1000; progress=true)

## Data Retrodiction
plot(; legend=false)
posterior_samples = sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
for p in eachrow(Array(posterior_samples))
    sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
    plot!(sol_p; alpha=0.1, color="#BBBBBB")
end

# Plot simulation and noisy observations.
scatter!([0:20], data'; color=[1 2])

## Using MAP instead
map_estimate = maximum_a_posteriori(model,initial_params=mean(chain).nt.mean)
StatsBase.coeftable(map_estimate)