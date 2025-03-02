using OrdinaryDiffEq, Random

# Check the working of helper functions

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
    dt = 0.1f0
    
    u0 = Float32.([0.44249296, 4.6280594])
    
    prob = ODEProblem(lotka_volterra, u0, tspan, p) 
    sol = solve(prob, Tsit5(), saveat=dt)
end 

# Test data sparsification
begin
    Random.seed!(1234)  # Ensure reproducibility
    t, X_sparse, mask = sparsify_data(sol, fraction=0.5)
    @test size(X_sparse) == size(sol)
    @test all(mask .== 0 .|| mask .== 1)
    @test sum(mask) < length(mask)
end

# Test batch mask 
begin 
    global_mask = mask
    batch_t = [0.3, 0.4, 0.5, 0.6]
    t0 = 0.0
    dt = 0.1
    expected_mask = [1, 0, 0, 0]
    
    batch_mask = get_mask_for_batch(batch_t, global_mask, t0, dt)
    @test batch_mask == expected_mask
end
