 

# Define a function to process each lineage
function process_lineage(df)
    df.Y = log.(df.length)
    df.time = 1:length(df[:,:x])
    df.cell = cumsum(df.div)
    df.z0 = vcat([c.Y[1]*ones(length(c.time)) for c in groupby(df,:cell)]...)
    df.S = cumsum(df.z0) + df.Y
    df.time = 1:length(df[:,:x])
    df.τ = vcat([(c.time[end]-c.time[1])*ones(length(c.time)) for c in groupby(df,:cell)]...)
    df.age = vcat([(c.time .- c.time[1]) for c in groupby(df,:cell)]...)
    df.T = df.age ./ df.τ
    return df
end