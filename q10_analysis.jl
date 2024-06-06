using JLD2

# Load the data
observations = load("data/observed_data_storm.jld2")["observed_data"]
X_data_0 = load("data/X_data_predict_0.jld2")["X_data"]
X_data_3 =  load("data/X_data_predict_3.jld2")["X_data"]
X_data_6 =  load("data/X_data_predict_6.jld2")["X_data"]
X_data_9 =  load("data/X_data_predict_9.jld2")["X_data"]
X_data_12 =  load("data/X_data_predict_12.jld2")["X_data"]
X_data_15 =  load("data/X_data_predict_15.jld2")["X_data"]
X_data_18 =  load("data/X_data_predict_18.jld2")["X_data"]
X_data_21 =  load("data/X_data_predict_21.jld2")["X_data"]
X_data_24 =  load("data/X_data_predict_24.jld2")["X_data"]
X_data_27 =  load("data/X_data_predict_27.jld2")["X_data"]
X_data_30 = load("data/X_data_predict_30.jld2")["X_data"]
X_data_33 = load("data/X_data_predict_33.jld2")["X_data"]
X_data_36 = load("data/X_data_predict_36.jld2")["X_data"]
X_data_39 = load("data/X_data_predict_39.jld2")["X_data"]

cutoffs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39] 
cutoff_times = s["t"][168 .- cutoffs] ./ 3600 .- 28

ilocs = [51, 101, 151, 199]
loc_names = ["Vlissingen", "Terneuzen", "Hansweert", "Bath"]
observations = observations[2:end, :]

# Ensemble means
ensemble_means = [
    mean(X_data_0, dims=3),
    mean(X_data_3, dims=3),
    mean(X_data_6, dims=3),
    mean(X_data_9, dims=3),
    mean(X_data_12, dims=3),
    mean(X_data_15, dims=3),
    mean(X_data_18, dims=3),
    mean(X_data_21, dims=3),
    mean(X_data_24, dims=3),
    mean(X_data_27, dims=3),
    mean(X_data_30, dims=3),
    mean(X_data_33, dims=3),
    mean(X_data_36, dims=3),
    mean(X_data_39, dims=3)
]

# Find the peak indices
peak_idcs = zeros(Int, length(ilocs))
for i ∈ eachindex(ilocs)
    peak_idcs[i] = 168 # or argmax(observations[i, :])
end

# Compute peak differences for each cutoff per location
peak_diffs = zeros(Float64, length(ilocs), length(cutoffs))

for i ∈ eachindex(ilocs)
    for j ∈ 1:length(cutoffs)
        peak_diffs[i, j] = observations[i, peak_idcs[i]] - ensemble_means[j][ilocs[i], peak_idcs[i]-1, 1]
    end
end

# Plot the peak differences
p1 = plot()
for i ∈ eachindex(ilocs)
    plot!(p1, cutoffs, peak_diffs[i, :], label=loc_names[i], xlabel="Cutoff [h]", ylabel="Error [m]", title="Peak Differences Based on Cut-Off Before Peak")
end
plot(p1, legend=:bottomright, title="Peak Differences Based on Cut-Off Before Peak", xticks=(cutoffs[1:2:end], string.(cutoff_times[1:2:end])))
savefig(p1, "figures/q10_peak_diff.png")


observations_real = load("data/observed_data_real.jld2")["observed_data"]

plot(s["h_left"])
plot!(observations_real[1, :])