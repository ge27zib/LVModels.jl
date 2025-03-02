using Flux, Optimisers, SciMLStructures

@doc """
    AbstractNDEModel

Supertype of the neural differential equation model defined in this package.
"""
abstract type AbstractNDEModel end

@doc """
    NDE{P,R,A,K} <: AbstractNDEModel

Model for setting up and training Neural Differential Equations.

# Fields:
- `p`: Parameter struct instance
- `prob`: DEProblem 
- `alg`: Algorithm to use for the `solve` command 
- `kwargs`: any additional keyword arguments that should be handed over (e.g. `sensealg`)

# Constructors 
- `NDE(prob; alg=Tsit5(), kwargs...)`
- `NDE(model::NDE; alg=model.alg, kwargs...)` remake the model with different kwargs and solvers

# Input / call 

An instance of the model is called with a trajectory pair `(t,x)` in `t` are the timesteps that NDE is integrated for and `x` is a trajectory `N x ... x N_t` in which `x[:, ... , 1]` is taken as the initial condition. 
"""
struct NDE{P,R,A,K} <: AbstractNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
end 

function NDE(prob; alg=Tsit5(), kwargs...)
    p = prob.p 
    return NDE{typeof(p),typeof(prob),typeof(alg),typeof(kwargs)}(p, prob, alg, kwargs)
end 

function (m::NDE)(X, p=m.p)
    (t, x) = X 
    Array(solve(remake(m.prob; tspan=(t[1], t[end]), u0=x[:,1], p=p), m.alg; saveat=t, m.kwargs...))
end

Flux.@functor NDE
Optimisers.trainable(m::NDE) = (p = m.p,)

@doc """
    mutable struct LearnableParams{T}

Learnable parameters while training the model.

# Fields:
- `θ`: Neural network weights
- `θ1`: Decay rates vector

# Constructor
- `LearnableParams(θ, θ1)`
"""
mutable struct LearnableParams{T}
    θ::T
    θ1::T
end

Flux.@functor LearnableParams

SciMLStructures.isscimlstructure(::LearnableParams) = true
ismutablescimlstructure(::LearnableParams) = true
SciMLStructures.hasportion(::SciMLStructures.Tunable, ::LearnableParams) = true

function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::LearnableParams)
    buffer = vcat(p.θ, p.θ1)
    repack = let p = p
        function repack(newbuffer)
            SciMLStructures.replace(SciMLStructures.Tunable(), p, newbuffer)
        end
    end
    return buffer, repack, false
end

function SciMLStructures.replace(::SciMLStructures.Tunable, p::LearnableParams, newbuffer)
    N = length(p.θ)
    l = length(p.θ1)
    @assert length(newbuffer) == N + l

    θ = newbuffer[1:N]  
    θ1 = newbuffer[N+1:N+l]

    return LearnableParams(θ, θ1)
end

function replace!(::SciMLStructures.Tunable, p::LearnableParams, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N + l 

    p.θ .= newbuffer[1:N]  
    p.θ1 .= newbuffer[N+1:N+l] 

    return p
end
