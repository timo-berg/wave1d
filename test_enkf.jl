using Plots

include("wave1d_enkf.jl")
include("wave1d_twin.jl")

x_data, series_data, observed_data, s = simulate_twin()
x_data = x_data[1:200, :]

n_ensembles = [10, 30, 50, 100, 200]

all_tide_rmse_time = []
all_velocity_rmse_time = []
all_tide_rmse_location = []
all_velocity_rmse_location = []

for n_ensemble in n_ensembles
    series_data, observed_data, X_data, s = simulate_enkf(n_ensemble)
    X_data = X_data[1:200, :, :]

    ensemble_mean = mean(X_data, dims=3)

    # Separate tides and velocities in x_data and ensemble_mean
    tides = x_data[1:2:end, :]
    velocities = x_data[2:2:end, :]
    ensemble_tides = ensemble_mean[1:2:end, :, 1]
    ensemble_velocities = ensemble_mean[2:2:end, :, 1]

    # RMSE time wise for tides and velocities
    tide_rmse_time = zeros(size(tides, 2))
    velocity_rmse_time = zeros(size(velocities, 2))
    for t_idx in axes(tides, 2)
        tide_rmse_time[t_idx] = sqrt(mean((tides[:, t_idx] .- ensemble_tides[:, t_idx]).^2))
        velocity_rmse_time[t_idx] = sqrt(mean((velocities[:, t_idx] .- ensemble_velocities[:, t_idx]).^2))
    end

    # RMSE location wise for tides and velocities
    tide_rmse_location = zeros(size(tides, 1))
    velocity_rmse_location = zeros(size(velocities, 1))
    for x_idx in axes(tides, 1)
        tide_rmse_location[x_idx] = sqrt(mean((tides[x_idx, :] .- ensemble_tides[x_idx, :]).^2))
        velocity_rmse_location[x_idx] = sqrt(mean((velocities[x_idx, :] .- ensemble_velocities[x_idx, :]).^2))
    end

    push!(all_tide_rmse_time, tide_rmse_time)
    push!(all_velocity_rmse_time, velocity_rmse_time)
    push!(all_tide_rmse_location, tide_rmse_location)
    push!(all_velocity_rmse_location, velocity_rmse_location)
end


# Plotting RMSE for tides
p1 = plot(title="Location wise RMSE for Tides", legend=:outertopright)
p2 = plot(title="Time wise RMSE for Tides", legend=:outertopright)
for i in 1:length(n_ensembles)
    plot!(p1, all_tide_rmse_location[i], label="n_ensemble = $(n_ensembles[i])")
    plot!(p2, all_tide_rmse_time[i], label="")
end

# Plotting RMSE for velocities
p3 = plot(title="Location wise RMSE for Velocities", legend=:outertopright)
p4 = plot(title="Time wise RMSE for Velocities", legend=:outertopright)
for i in 1:length(n_ensembles)
    plot!(p3, all_velocity_rmse_location[i], label="n_ensemble = $(n_ensembles[i])")
    plot!(p4, all_velocity_rmse_time[i], label="")
end

# Combine plots
plot(plot(p1, p2, layout=(2,1)), plot(p3, p4, layout=(2,1)), layout=(2,1))