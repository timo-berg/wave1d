using JLD2

# Load the data
observations = load("data/observed_data_storm.jld2")["observed_data"]
X_data_0 =  load("data/X_data_predict_0.jld2")["X_data"]
X_data_10 = load("data/X_data_predict_10.jld2")["X_data"]
X_data_20 = load("data/X_data_predict_20.jld2")["X_data"]
X_data_30 = load("data/X_data_predict_30.jld2")["X_data"]
X_data_40 = load("data/X_data_predict_40.jld2")["X_data"]
X_data_50 = load("data/X_data_predict_50.jld2")["X_data"]
X_data_60 = load("data/X_data_predict_60.jld2")["X_data"]
X_data_70 = load("data/X_data_predict_70.jld2")["X_data"]
X_data_80 = load("data/X_data_predict_80.jld2")["X_data"]
X_data_90 = load("data/X_data_predict_90.jld2")["X_data"]
X_data_100 = load("data/X_data_predict_100.jld2")["X_data"]
X_data_110 = load("data/X_data_predict_110.jld2")["X_data"]
X_data_120 = load("data/X_data_predict_120.jld2")["X_data"]
X_data_130 = load("data/X_data_predict_130.jld2")["X_data"]


cutoffs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
cutoff_times = round.(s["t"][168 .- cutoffs] ./3600, digits=2)

ilocs = [51, 101, 151, 199]
loc_names = ["Vlissingen", "Terneuzen", "Hansweert", "Bath"]
observations=observations[2:end, :]

# ensemble means
ensemble_mean_0 = mean(X_data_0, dims=3)
ensemble_mean_10 = mean(X_data_10, dims=3)
ensemble_mean_20 = mean(X_data_20, dims=3)
ensemble_mean_30 = mean(X_data_30, dims=3)
ensemble_mean_40 = mean(X_data_40, dims=3)
ensemble_mean_50 = mean(X_data_50, dims=3)
ensemble_mean_60 = mean(X_data_60, dims=3)
ensemble_mean_70 = mean(X_data_70, dims=3)
ensemble_mean_80 = mean(X_data_80, dims=3)
ensemble_mean_90 = mean(X_data_90, dims=3)
ensemble_mean_100 = mean(X_data_100, dims=3)
ensemble_mean_110 = mean(X_data_110, dims=3)
ensemble_mean_120 = mean(X_data_120, dims=3)
ensemble_mean_130 = mean(X_data_130, dims=3)

# find the peak idcs
peak_idcs = zeros(Int, length(ilocs))
for i ∈ eachindex(ilocs)
    peak_idcs[i] = 168#argmax(observations[i, :])
end

# compute peak differences for each cutoff per location
peak_diffs = zeros(Float64, length(ilocs), length(cutoffs))

for i ∈ eachindex(ilocs)
    peak_diffs[i, 1] = observations[i,peak_idcs[i]] - ensemble_mean_0[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 2] = observations[i,peak_idcs[i]] - ensemble_mean_10[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 3] = observations[i,peak_idcs[i]] - ensemble_mean_20[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 4] = observations[i,peak_idcs[i]] - ensemble_mean_30[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 5] = observations[i,peak_idcs[i]] - ensemble_mean_40[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 6] = observations[i,peak_idcs[i]] - ensemble_mean_50[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 7] = observations[i,peak_idcs[i]] - ensemble_mean_60[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 8] = observations[i,peak_idcs[i]] - ensemble_mean_70[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 9] = observations[i,peak_idcs[i]] - ensemble_mean_80[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 10] = observations[i,peak_idcs[i]] - ensemble_mean_90[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 11] = observations[i,peak_idcs[i]] - ensemble_mean_100[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 12] = observations[i,peak_idcs[i]] - ensemble_mean_110[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 13] = observations[i,peak_idcs[i]] - ensemble_mean_120[ilocs[i], peak_idcs[i]-1, 1]
    peak_diffs[i, 14] = observations[i,peak_idcs[i]] - ensemble_mean_130[ilocs[i], peak_idcs[i]-1, 1]
end

# plot the peak differences
p1 = plot()
for i ∈ eachindex(ilocs)
    plot!(p1, cutoffs, peak_diffs[i,:], label=loc_names[i], xlabel="Cutoff [h]", ylabel="Waterlevel [m]")
end
plot(p1, legend=:bottomright, title="Peak Differences base on cut-off before peak", xticks=(cutoffs[1:2:end], cutoff_times[1:2:end]))
savefig(p1, "/figures/q10_peak_diff.png")