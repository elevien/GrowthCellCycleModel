# simulates the model for a range of bθ and bz

using DataFrames,StatsBase,Random,Distributions,CSV
include("model.jl")


dt = 0.01
T = 9000
ϕ0 = log(2)
σ1 = 0.001
σ2 = 0.01
α = 0.5

# Scan ranges
b_range = collect(-2:0.1:2) 
a_range = collect(0.5:0.5:5.0) 



dfs_cells = []


for a in a_range
    for b in b_range
        # Set model-specific parameters
        init = [0.0,0.0,0.0,0.0]
        params = (a,b,ϕ0,α,σ1,σ2)
        df = make_sim_df_ratio_model(init, params, dt, T);
        df.y = df.z .+ log(2) .*df.θ;

        dfcell = combine(groupby(df, :cell),
            :z => first => :z0,
            :y => (x -> x[end] - x[1]) => :ϕ,
            :time => (x -> x[end] - x[1]) => :τ,
            :x1 => mean => :x1,
            :x2 => mean => :x2,
        )
        dfcell.λ .= dfcell.ϕ ./ dfcell.τ
        dfcell.a .= a
        dfcell.b .= b
        dfcell.isphys .=  isphysical(df)
        push!(dfs_cells, dfcell)
        print("Processed a: $a, b: $b\n")
    end
end

df_cells = vcat(dfs_cells...)

# Save
CSV.write("./output/ratio_data_cells.csv",df_cells)


