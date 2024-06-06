using JLD2
using Plots
using Statistics
using LsqFit

function rmse(a, b)
    return sqrt(mean((a .- b) .^ 2))
end

# Question 7
ilocs = [1, 51, 101, 151, 199]
loc_names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
# data loading
n_ensembles = [5, 10, 20, 50, 100]

observed_data = load("data/observed_data_sim.jld2")["observed_data"]

X_data_5 = load("data/X_data_sim_ass_5.jld2")["X_data"]
X_data_10 = load("data/X_data_sim_ass_10.jld2")["X_data"]
X_data_20 = load("data/X_data_sim_ass_20.jld2")["X_data"]
X_data_50 = load("data/X_data_sim_ass_50.jld2")["X_data"]
X_data_100 = load("data/X_data_sim_ass_100.jld2")["X_data"]

# ensemble means
ensemble_mean_5 = mean(X_data_5, dims=3)
ensemble_mean_10 = mean(X_data_10, dims=3)
ensemble_mean_20 = mean(X_data_20, dims=3)
ensemble_mean_50 = mean(X_data_50, dims=3)
ensemble_mean_100 = mean(X_data_100, dims=3)

# make a plot
rmse_sim_ass = zeros(Float64, length(ilocs), length(n_ensembles))

for i ∈ eachindex(ilocs)
    rmse_sim_ass[i, 1] = rmse(ensemble_mean_5[ilocs[i], :, :], observed_data[i, 1:end-1])
    rmse_sim_ass[i, 2] = rmse(ensemble_mean_10[ilocs[i], :, :], observed_data[i, 1:end-1])
    rmse_sim_ass[i, 3] = rmse(ensemble_mean_20[ilocs[i], :, :], observed_data[i, 1:end-1])
    rmse_sim_ass[i, 4] = rmse(ensemble_mean_50[ilocs[i], :, :], observed_data[i, 1:end-1])
    rmse_sim_ass[i, 5] = rmse(ensemble_mean_100[ilocs[i], :, :], observed_data[i, 1:end-1])
end

# p1 = plot()

# for i ∈ eachindex(n_ensembles)
#     plot!(p1, rmse_sim_ass[:,i], label="n=$(n_ensembles[i])", xticks=(collect(1:length(ilocs)), loc_names))
# end

# p2 = plot()

# for i ∈ eachindex(ilocs)
#     plot!(p2, rmse_sim_ass[i,:], label="$(loc_names[i])", xticks=(collect(1:length(n_ensembles)), n_ensembles))
# end

# plot(p1, p2, layout=(2,1))

param_mat = zeros(Float64, length(ilocs), 3)
plots = []

for loc_idx ∈ eachindex(ilocs)
    # Log-Log Plot
    p = plot(n_ensembles, rmse_sim_ass[loc_idx, :], label="Measured", xlabel="n", ylabel="RMSE", title=loc_names[loc_idx])

    m(t, p) = p[3] .* t .^ p[2] .+ p[1]
    p0 = [0, -0.5, 1]

    fit = curve_fit(m, n_ensembles, rmse_sim_ass[loc_idx, :], p0)

    pred(n) = fit.param[3] * n^fit.param[2] + fit.param[1]

    predictions = pred.(n_ensembles)

    plot!(p, n_ensembles, predictions, label="Fitted")

    param_mat[loc_idx, :] = fit.param

    push!(plots, p)
end

total_plot = plot(plots..., size=(1200, 600), dpi=1000, legend=:bottomright)
savefig(total_plot, "figures/q7_convergencec_fit.png")

round.(param_mat, digits=4)