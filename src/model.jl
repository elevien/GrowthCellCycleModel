using LinearAlgebra
using Random
using DataFrames

function make_sim_df(init, params, dt, T)
    A, by, b̃θ, uz, nux, Cθ, Γ, ν0 = params

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
        θ += (ν0 - dot(x, nux) + q) * dt

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


function coarse_grained1D(dfcell::DataFrame, l::Int)
    n = nrow(dfcell)

    Y = dfcell.ϕ[l+1:end]
    X = zeros(n-l, 2l+1)
    # standardize the data
    #Y = (Y .- mean(Y)) ./ std(Y)    
    # Fill in the X matrix  

    for j in 1:l # each j represents a lag and we are filling in ALL rows for that lag
        X[:, j] = dfcell.ϕ[l-j+1:end-j]
        X[:, l + j+1] = dfcell.z0[l-j+1:end-j]
    end
    X[:, l+1] = dfcell.z0[l+1:end];
     # Add intercept column
    # standardize X 
    #X = (X .- mean(X, dims=2)) ./ std(X, dims=2)
    X_aug = hcat(ones(size(X, 1)), X)
    
    # Solve normal equations via least squares
    β = X_aug \ Y  # (X'X)β = X'Y
    βϕ = β[2:l+1]
    βz = β[l+2:end]
    return βϕ,βz, X_aug, Y

end


function lagged_covariances(df::DataFrame, L::Int)
    ϕ = df.ϕ
    λ = df.λ
    y0 = df.z0
    n = length(ϕ)
    Σ = Matrix{Float64}[]

    for j in 0:L
        # Align sequences for lag j
        ϕ_t = ϕ[(j+1):end]
        ϕ_tmj = ϕ[1:(end-j)]
        λ_t = λ[(j+1):end]
        λ_tmj = λ[1:(end-j)]
        y_t = y0[(j+1):end]
        y_tmj = y0[1:(end-j)]

        # Compute lagged covariances
        Σj = [
            cov(λ_tmj, λ_t)  cov(ϕ_tmj, λ_t) cov(λ_tmj, y_t);
            cov(λ_tmj, ϕ_t)  cov(ϕ_tmj, ϕ_t) cov(ϕ_tmj, y_t);
            cov(y_tmj, λ_t)  cov(y_tmj, ϕ_t)  cov(y_tmj, y_t)
        ]
        push!(Σ, Σj)
    end

    return Σ
end


function build_model2d(a1,a2,ω0,q,σ,c)
    A = [-a1 0.0; 0.0 -a2]
    Γ = [σ 0.0; 0.0 σ]
    uz = [1.0,-1.0]
    nux = [c/log(2),1.0/log(2)]
    Cθ = zeros(2,2)
    Cθ[1,2] = c/log(2)^2/2
    Cθ[2,1] = c/log(2)^2/2
    b̃θ = zeros(2)
    
    by = ω0^2 .* [-q,1-q]
    #b̃θ = log(2)*by
    ν0 = 1/log(2)
    
    params = (A,by,b̃θ,uz,nux,Cθ,Γ,ν0)
    return params
end
