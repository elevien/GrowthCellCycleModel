# simulates the model for a range of bθ and bz

using DataFrames,StatsBase,Random,Distributions,CSV
include("model.jl")


# Data containers

dt = 0.01
T = 10000
init = [0.0, 0.0, 0.0, 0.0]
vz = 0.001
ν0 = 1/log(2)



q_range = collect(0.0:0.05:1.0)
ω_range = collect(0.01:0.5:10.0)
η_range = collect(0.1:1.:10.0)  
c = 0.
#dfs = []
dfs_cells = []


for η in η_range
    for om in ω_range
        for q in q_range
            # solve alpha_SHO(a, om, 1/log(2), η) = 0.5 for η
            a = om*η
            
            σ = sqrt(vz*a*om^2) # variance of noise in z
            params = build_model2d(a,a,om,q,σ,c)
            df = make_sim_df(init, params, dt, T)

            df.y = df.z .+ log(2) .* df.θ
            dfcell_sim = combine(groupby(df, :cell),
                    :z => first => :z0,
                    :y => (x -> x[end] - x[1]) => :ϕ,
                    :time => (x -> x[end] - x[1]) => :τ,
                    :x1 => mean => :x1,
                    :x2 => mean => :x2)
            dfcell_sim.λ = dfcell_sim.ϕ ./dfcell_sim.τ
            dfcell_sim.q .= q
            df.q .= q
            df.ω0 .= om
            df.η .= η
            dfcell_sim.ω0 .= om
            dfcell_sim.η .= η
            dfcell_sim.c .= c
            dfcell_sim.isphys .=  isphysical(df)
            #push!(dfs,df)
            push!(dfs_cells,dfcell_sim)
            println("Processed η: $η, ω0: $om, q: $q")
        end
    end
end


#dfs = vcat(dfs...);
dfs_cells = vcat(dfs_cells...);

# Save
CSV.write("./output/sho_data_cells_2d.csv",dfs_cells)


