using JLD2
using Plots
using Statistics

function rmse(a, b)
    return sqrt(mean((a .- b) .^ 2)) / length(a)
end

### Question 6
ilocs = [1, 51, 101, 151, 199] #  , 2, 52, 102, 152
loc_names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
# data loading
observed_data = load("data/observed_data.jld2")["observed_data"]
observed_data_real = load("data/observed_data_real.jld2")["observed_data"]
X_data_enkf = load("data/X_data_enkf.jdl2")["X_data"]
X_data_no_ass = load("data/X_data_no_ass.jdl2")["X_data"]
X_data_sim_ass = load("data/X_data_sim_ass.jdl2")["X_data"]

# Enkf has reduced spread for every measurement
# sim_ass has reduced spread for every measurement
# no_ass has no spread reduction
spread_enkf = zeros(Float64, length(ilocs))
spread_sim_ass = zeros(Float64, length(ilocs))
spread_no_ass = zeros(Float64, length(ilocs))


for i ∈ eachindex(ilocs)
    spread_enkf[i] = mean(std(X_data_enkf[ilocs[i], :, :], dims=2)) # spread across ensemlbes and then time average
    spread_sim_ass[i] = mean(std(X_data_sim_ass[ilocs[i], :, :], dims=2))
    spread_no_ass[i] = mean(std(X_data_no_ass[ilocs[i], :, :], dims=2))
end

p1 = plot(spread_enkf, label="Real Assimilation", xticks=(collect(1:length(ilocs)), loc_names), legend=:right, dpi=1000)
plot!(p1, spread_sim_ass, label="Synthetic Assimilation")
plot!(p1, spread_no_ass, label="No Assimilation")
title!(p1, "Standard deviation of the ensemble")
savefig(p1, "figures/q6_std_ensemble.png")

# RMSE between ensemble mean and obeserved_data is lower in sim_ass than in enkf
ensemble_mean_enkf = mean(X_data_enkf, dims=3)
ensemble_mean_sim_ass = mean(X_data_sim_ass, dims=3)

rmse_enkf = zeros(Float64, length(ilocs))
rmse_sim_ass = zeros(Float64, length(ilocs))

for i ∈ eachindex(ilocs)
    rmse_enkf[i] = rmse(ensemble_mean_enkf[ilocs[i], :, :], observed_data_real[i, 2:end])
    rmse_sim_ass[i] = rmse(ensemble_mean_sim_ass[ilocs[i], :, :], observed_data[i, 2:end])
end

p2 = plot(rmse_enkf, label="Real Assimilation", xticks=(collect(1:length(ilocs)), loc_names), legend=:right, dpi=1000)
plot!(p2, rmse_sim_ass, label="Synthetic Assimilation")
title!(p2, "RMSE of the ensemble mean and measurement data")
savefig(p2, "figures/q6_rmse_ensemble.png")

# obeserved_data lies between min(ensemble) and max(ensemble) of sim_ass
is_between_matrix_enkf = zeros(Float64, (length(ilocs), 288))
is_between_matrix_sim_ass = zeros(Float64, (length(ilocs), 288))

tolerance = 0.1

for loc_idx ∈ eachindex(ilocs)
    ensemble_max_sim_ass = maximum(X_data_sim_ass[ilocs[loc_idx], :, :], dims=2) .+ tolerance
    ensemble_min_sim_ass = minimum(X_data_sim_ass[ilocs[loc_idx], :, :], dims=2) .- tolerance

    ensemble_max_enkf = maximum(X_data_enkf[ilocs[loc_idx], :, :], dims=2) .+ tolerance
    ensemble_min_enkf = minimum(X_data_enkf[ilocs[loc_idx], :, :], dims=2) .- tolerance

    is_between_matrix_enkf[loc_idx, :] = ensemble_min_enkf .< observed_data_real[loc_idx, 2:end] .< ensemble_max_enkf
    is_between_matrix_sim_ass[loc_idx, :] = ensemble_min_sim_ass .< observed_data[loc_idx, 2:end] .< ensemble_max_sim_ass
end

p3 = heatmap(is_between_matrix_enkf,
    color=:blues,
    aspect_ratio=:auto,
    yticks=(collect(1:length(ilocs)), loc_names),
    title="Real Assimilation")

p4 = heatmap(is_between_matrix_sim_ass,
    color=:blues,
    aspect_ratio=:auto,
    xlabel="Time Step",
    yticks=(collect(1:length(ilocs)), loc_names),
    title="Synthetic Assimilation")

p5 = plot(p3, p4, layout=(2,1), dpi=1000)
savefig(p5, "figures/q6_ensemble_capture.png")

p6 = plot(sum(is_between_matrix_enkf, dims=2)./2.88, label="Real Assimilation", ylabel="% covered", xticks=(collect(1:length(ilocs)), loc_names), dpi=1000)
plot!(p6, sum(is_between_matrix_sim_ass, dims=2)./2.88, label="Synthetic Assimilation")
title!("Ensemble cover of measurement")
savefig(p6, "figures/q6_ensemble_cover_summary.png")