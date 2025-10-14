# simulates the model for a range of bθ and bz

using DataFrames,StatsBase,Random,Distributions,CSV
include("model.jl")


# Initialize parameters
d = 2
A = [-1.0 0.0; 0.0 -1.0]
Γ = [0.025 0.0; 0.0 0.0]

dt = 0.01
T = 900
ν0 = 1/log(2) - 0.1


# Define range for uθ
uθrange = collect(-5:0.15:5)
bθrange = collect(-5.0:0.15:5.0)


# Initialize arrays to store dataframes
dfs = []
dfs_cells = []

# Loop over uθ values
for i in eachindex(uθrange)
    for j in eachindex(bθrange)
        println("uθ: $(uθrange[i]), bθ: $(bθrange[j])")


        uθ_vect = [uθrange[i],0.0]
        uz_vect = [1.0,0.0]
        uy_vect = uz_vect .+ uθ_vect*log(2)

        B = [-1.0 bθrange[j]; 0.0 0.0]
        params = (A,B,uz_vect,uθ_vect,Γ,ν0)
        init = [0.0, 0.0, 0.0, 0.0]
        
        # Generate simulation dataframe
        df = make_sim_df(init, params, dt, T)
        df.uθ = uθrange[i]*ones(length(df.z))
        df.bθ = bθrange[j]*ones(length(df.z))
        df.y = df.z + df.θ*log(2)

        
        
        # Generate cell-level statistics
        dfcell = combine(groupby(df,:cell),
            :z => (x -> x[1]) => :z0,
            :y => (x -> x[end] - x[1]) => :ϕ,
            :time => (x -> x[end]-x[1]) => :τ,
            :x1 => (x -> mean(x)) => :x1,
            :x2 => (x -> mean(x)) => :x2,
            :nux => (x -> x[1]) => :nux,
            :bθ => (x -> x[1]) => :bθ
        )
        dfcell.λ = dfcell.ϕ ./ dfcell.τ
        dfcell = dfcell[1:end-1,:]

        push!(dfs, df[df.time .> T - 50,:])
        push!(dfs_cells, dfcell)
    end
end



# Save the data
CSV.write("./../output/fig3_data_all.csv",vcat(dfs...))
CSV.write("./../output/fig3_data_cells.csv",vcat(dfs_cells...))