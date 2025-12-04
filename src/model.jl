using LinearAlgebra
using Random
using DataFrames

function make_sim_df(init, params, dt, T)
    A, by, b̃θ, uz, nux, Cθ, Γ, ν0 = params

    # note that Cθ is a quadratic term which is zero for the paper
    d = length(init) - 2
    x = copy(@view init[1:d])              # current state (mutable)
    z = init[d+1]
    θ = init[d+2]

    log2 = log(2)
    nsteps = Int(floor(T / dt))            # number of Euler steps
    cell = 0.0

    # Preallocate output: one row per step (time starts at dt)
    R = Matrix{Float64}(undef, nsteps, d + 3)  # [x..., z, θ, cell]

    # Scratch buffers (reused every step)
    Cθx = similar(x)                       # holds Cθ * x 
    Ax  = similar(x)                       # holds A  * x
    ξ    = Vector{Float64}(undef, size(Γ, 2))
    Γξ   = similar(x)                      # holds Γ * ξ

    √dt = sqrt(dt)

    @inbounds for t in 1:nsteps
        # q = x' * Cθ * x, computed without allocations
        mul!(Cθx, Cθ, x)                   # Cθx = Cθ * x
        q = dot(x, Cθx)

        # Update z and θ
        z += (dot(x, uz) + (1 - ν0 * log2) - log2 * q) * dt
        θ += (ν0 + dot(x, nux) + q) * dt

        # Update x
        mul!(Ax, A, x)                     # Ax = A * x
        randn!(ξ)                          # ξ ~ N(0, I)
        mul!(Γξ, Γ, ξ)                     # Γξ = Γ * ξ

        @. x = x + Ax * dt + by * (z * dt) + b̃θ * (θ * dt) + Γξ * √dt

        # Threshold/reset for θ (x update uses pre-reset θ, matching original)
        if θ ≥ 1
            θ = 0.0
            cell += 1.0
        end

        # Write row t
        R[t, 1:d]      = x
        R[t, d+1]      = z
        R[t, d+2]      = θ
        R[t, d+3]      = cell
    end

    cols = vcat([Symbol(:x, i) for i in 1:d], [:z, :θ, :cell])
    df = DataFrame(R, cols)
    df.time = (1:nsteps) .* dt            # matches original (starts at dt, not 0)
    return df
end



function make_sim_df_ratio_model(init, params, dt, T)

    a,b,ϕ0,α,σ1,σ2 = params
    # Initialize variables
    
    d = length(init)-2
    x = init[1:d]
    z,θ = init[d+1],init[d+2]

    cell = 0.0
    times = 0:dt:T

    # Storage for results
    results = [vcat(x,[z, θ, cell])]
    for _ in times
        x = results[end][1:d]
        z, θ, cell = results[end][d+1:end]
        # Compute derivatives
        # x[2] is the ratio of the two components psik
        z = z + (1 + x[1]) * (1 - log(2)/(ϕ0 + x[2])) * dt
        θ = θ + (1 + x[1]) / (ϕ0 + x[2])*dt  # Update Theta
        x[1] = x[1] - a*x[1]*dt - b *z *dt +  σ1 * randn() * sqrt(dt)

        # Handle Theta reset
        if θ ≥ 1
            θ = 0
            cell += 1
            x[2] = -α*z +  σ2*randn()
            push!(results, vcat(x,[z, 0, cell]))
        else
            push!(results, vcat(x,[z, θ, cell]))
        end
    end
    columns = vcat([Symbol(:x, i) for i in 1:d],[:z,:θ,:cell])
    df = DataFrame(hcat(results...)', columns)
    df.time = collect(1:length(df.cell))*dt
    return df
end 


function isphysical(df::DataFrame)
    for i in unique(df.cell)[1:end-1]
        d = df[df.cell .== i,:]
        d.y = d.z .+ log(2)*d.θ
        dt = d.time[2] - d.time[1]
        dydt = diff(d.y) ./ dt
        dθdt = diff(d.θ) ./ dt
        if (min(dydt...).<0) .|| (min(dθdt...).<0)
            return 0
        end
    end
    return 1
end


function transform_directions(uy,nux,by,bθ)
    # determine other vectors
    uz = uy .- nux*log(2)
    bz = by 
    b̃θ = bθ .+ bz*log(2)
    return uz,bz,bz,b̃θ
end


function fitarl(Y, l::Int)
    n = length(Y)
    # Y is 1D array of observations
    # l is the order of the AR model
    
    X = zeros(n - l, l)
    for i in 1:(n-l)
        X[i, :] = Y[i:l]
    end
    Y_target = Y[(l+1):end]
    
    # Solve for coefficients using least squares
    coeffs = X \ Y_target
    return coeffs
end





function build_model2d(a1,a2,ω0,q,σ)
    A = [-a1 0.0; 0.0 -a2]
    Γ = [σ 0.0; 0.0 σ]
    uz = [1.0,-1.0]
    nux = [0.0,1.0/log(2)]
    Cθ = zeros(2,2)
    b̃θ = zeros(2)
    
    by = ω0^2 .* [-q,1-q]
    ν0 = 1/log(2)
    
    params = (A,by,b̃θ,uz,nux,Cθ,Γ,ν0)
    return params
end
