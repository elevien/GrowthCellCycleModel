# simulates the model for a range of bθ and bz

using DataFrames,StatsBase,Random,Distributions,CSV
include("model.jl")


dt = 0.01
T = 10000
ν0 = 1/log(2)
vz = 0.001

# Scan ranges
ω_range = collect(0.25:0.25:6.5)   # ω₀
η_range = [0.5,1.5,4.0]   # diagonal values of A
nux_range = collect(-2.0:0.25:2.0)  # uₜ

# Data containers

dfs_cells = []

for ω0 in ω_range
    for η in η_range
        for nux in nux_range
            # Set model-specific parameters
            bz = -ω0^2
            a = ω0*η
            σ = sqrt(vz*a*ω0^2) # variance of noise in z
            A = [-a 0.0; 0.0 -a]
            Γ = [σ 0.0; 0.0 σ]
            uz = [1.0,0.0]
            bz = [bz, 0.0]
            nuxv = zeros(2)
            nuxv[1] = nux
            Cθ = zeros(2,2)
            b̃θ = zeros(2)
            params = (A,bz,b̃θ,uz,nuxv,Cθ,Γ,ν0)
            init = [0.0, 0.0, 0.0, 0.0]
            df = make_sim_df(init, params, dt, T)
            df.ω0 .= ω0
            df.η .= η
            df.y = df.z .+ log(2)*df.θ  # y = z + log(2) * θ

            # Cell-level stats
            dfcell = combine(groupby(df, :cell),
                :z => first => :z0,
                :y => (x -> x[end] - x[1]) => :ϕ,
                :time => (x -> x[end] - x[1]) => :τ,
                :x1 => mean => :x1,
                :x2 => mean => :x2,
            )
            dfcell.λ .= dfcell.ϕ ./ dfcell.τ
            dfcell.ω0 .= ω0
            dfcell.η .= η
            dfcell.nux .= nux
            dfcell.isphys .=  isphysical(df)
            dfcell.a .= a
            push!(dfs_cells, dfcell)
            println("Processed ω0: $ω0, η: $η, nux: $nux")
        end
    end
end
df_cells = vcat(dfs_cells...)

# Save
CSV.write("./output/sho_data_cells_1d.csv",df_cells)


