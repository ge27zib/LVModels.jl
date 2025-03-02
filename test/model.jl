using OrdinaryDiffEq, SciMLSensitivity, Flux, Optimisers

# test with a Lotka Volterra system, adjusted from scripts/lv_2d.jl 
# we just test if everything compiles and runs without errors 

begin
    N_WEIGHTS = 5
    dt = 0.1f0
end 

begin 
    function lotka_volterra(u,p,t)
        α, β, γ, δ = p 
        [α*u[1] - β*u[1]*u[2], -γ*u[2] + δ*u[1]*u[2]]
    end
    
    α = 1.3
    β = 0.9
    γ = 0.8
    δ = 1.8
    p = [α, β, γ, δ] 
    tspan = (0.,5.)
    
    u0 = Float32.([0.44249296, 4.6280594])
    
    prob = ODEProblem(lotka_volterra, u0, tspan, p) 
    sol = solve(prob, Tsit5(), saveat=dt)
end 

t = Float32.(0:dt:5.)
train = [(t, Float32.(Array(sol(t))))]

nn = Chain(Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2))
θ, U_re = Optimisers.destructure(nn)
p_ln = LearnableParams(θ, Float32[0.0]) # Learnable decay parameter

function neural_ode(u, p_ln::LearnableParams, t)
    x, y = u
    U = U_re(p_ln.θ)(u)
    [1.3f0 * x + U[1], -1.8f0 * y + U[2]]
end

model = NDE(ODEProblem(neural_ode, u0, (0f0, dt), p_ln); reltol=1f-5, dt=dt)

model(train[1])

loss(m, x, y) = Flux.mse(m(x),y) 
loss(model, train[1], train[1][2]) 

# check if the gradient works 
g = gradient(model) do m
    loss(m, train[1], train[1][2])
end
pgrad1 = g[1][:p][1]
pgrad2 = g[1][:p][2]

# do a check that the gradient is nonzero, noninf and nonnothing
@test sum(isnan.(pgrad1)) == 0
@test sum(isnan.(pgrad2)) == 0
@test sum(isinf.(pgrad1)) == 0 
@test sum(isinf.(pgrad2)) == 0 
@test sum(isnothing.(pgrad1)) == 0 
@test sum(isnothing.(pgrad2)) == 0 
