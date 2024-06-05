using Latexify
using DataFrames
using JLD2
using Statistics
using Peaks
using Plots

struct stats #statistics of peaks
    mean::Float64
    std::Float64
end

function compute_peak_statistic(series_data, observed_data, s)
    nseries = length(s["loc_names"])

    # Find the peaks in the series data
    for i = 1:5
        amplitude_error, timing_error = peak_statistic(series_data[i, :], observed_data[i, :])
        println("Location: $(s["loc_names"][i])")
        println("Amplitude error:\n μ: $(amplitude_error.mean), σ: $(amplitude_error.std)")
        println("Mean timing error [min]:\n μ:$(timing_error.mean * s["dt"] / 60), σ: $(timing_error.std * s["dt"] / 60)")
    end
end

function compute_rmse_bias(series_data, observed_data)
    error = series_data .- observed_data

    rmse = sqrt(mean(error .^ 2))
    bias = mean(error)

    return (rmse, bias)
end

function keep_positive(idcs, arr)
    return idcs[arr.>=0], arr[arr.>=0]
end

function keep_negative(idcs, arr)
    return idcs[arr.<=0], arr[arr.<=0]
end


function peak_statistic(model_data, observed_data)
    peak_min_tolerance = 7

    (peak_model_idcs, peak_model_vals, _) = findmaxima(model_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)
    (peak_obs_idcs, peak_obs_vals, _) = findmaxima(observed_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)

    (trough_model_idcs, trough_model_vals, _) = findminima(model_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)
    (trough_obs_idcs, trough_obs_vals, _) = findminima(observed_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)

    # Error if the number of peaks in the model and observations do not match
    if length(peak_model_idcs) > length(peak_obs_idcs)
        peak_model_idcs = peak_model_idcs[1:length(peak_obs_idcs)]
        peak_model_vals = peak_model_vals[1:length(peak_obs_idcs)]
    elseif length(peak_model_idcs) < length(peak_obs_idcs)
        peak_obs_idcs = peak_obs_idcs[1:length(peak_model_idcs)]
        peak_obs_vals = peak_obs_vals[1:length(peak_model_idcs)]
    end

    if length(trough_model_idcs) > length(trough_obs_idcs)
        trough_model_idcs = trough_model_idcs[1:length(trough_obs_idcs)]
        trough_model_vals = trough_model_vals[1:length(trough_obs_idcs)]
    elseif length(trough_model_idcs) < length(trough_obs_idcs)
        trough_obs_idcs = trough_obs_idcs[1:length(trough_model_idcs)]
        trough_obs_vals = trough_obs_vals[1:length(trough_model_idcs)]
    end


    # Compute the error statistics
    amplitude_errors = [abs.(peak_model_vals .- peak_obs_vals); abs.(trough_model_vals .- trough_obs_vals)]
    timing_errors = [abs.(peak_model_idcs .- peak_obs_idcs); abs.(trough_model_idcs .- trough_obs_idcs)] .* 10 # Convert to minutes

    amplitude_error = stats(round(mean(amplitude_errors), digits=5), round(std(amplitude_errors), digits=5))
    timing_error = stats(round(mean(timing_errors), digits=5), round(std(timing_errors), digits=5))

    # Modify this for more advanced error statistics
    return amplitude_error, timing_error
end


# Load the data
observed_data_chill = load("data/observed_data_real.jld2")["observed_data"]
X_data_no_ass = load("data/X_data_no_ass.jdl2")["X_data"]

ilocs = [1, 51, 101, 151, 199]
loc_names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
ensemble_mean_no_ass = mean(X_data_no_ass[ilocs, :, :], dims=3)





# Stats between ensembles and observations storm
error_stats = DataFrame(
    Location=["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"],
    RMSE=zeros(Float64, 5),
    Bias=zeros(Float64, 5),
    Amplitude_Mean=zeros(Float64, 5),
    Amplitude_Std=zeros(Float64, 5),
    Timing_Mean=zeros(Float64, 5),
    Timing_Std=zeros(Float64, 5)
)



for i = 1:5
    # smoothing the data
    ensemble_mean_smooth = filtfilt(ones(10) / 10, ensemble_mean_no_ass[i, :, 1])
    observed_data_smooth = filtfilt(ones(10) / 10, observed_data_storm[i, 2:end])

    amplitude_error, timing_error = peak_statistic(ensemble_mean_smooth, observed_data_smooth)
    rmse, bias = compute_rmse_bias(ensemble_mean_smooth, observed_data_smooth)

    error_stats[i, :RMSE] = round(rmse, digits=2)
    error_stats[i, :Bias] = round(bias, digits=2)
    error_stats[i, :Amplitude_Mean] = round(amplitude_error.mean, digits=2)
    error_stats[i, :Amplitude_Std] = round(amplitude_error.std, digits=2)
    error_stats[i, :Timing_Mean] = round(timing_error.mean, digits=2)
    error_stats[i, :Timing_Std] = round(timing_error.std, digits=2)
end

error_stats_no_ass = deepcopy(error_stats)


latexify(error_stats_no_ass, env=:table, latex=false)

p = plot(xlabel="Time [h]", ylabel="Waterlevel [m]", title="Standard deviation of the ensemble accross time")
for i ∈ eachindex(ilocs)
    variances = std(X_data_no_ass[ilocs[i], :, :], dims=2)
    variances = filtfilt(ones(5) / 5, variances)
    plot!(p, 1/3600 .* s["t"], variances, label=loc_names[i])
end
plot(p, legend=:outerbottomright, legend_title="Location", size=(600,400), dpi=1000)
savefig(p, "figures/q4_std_ensemble_time.png")

times = Int.(round.(LinRange(2, 288, 4)))

times = [1, 3, 5, 7, 10, 15, 20, 35, 50, 100, 250]
colors = cgrad(:RdYlBu, times, rev=true)  # Use the Viridis color scale


p = plot(xlabel="Distance [m]", ylabel="Waterlevel [m]", title="Standard deviation of the ensemble accross space")
for i ∈ eachindex(times)
    variances = std(X_data_no_ass[1:2:end, times[i], :], dims=2)
    variances = filtfilt(ones(5) / 5, variances)
    plot!(p, variances, label=1/60 * s["t"][times[i]], color=colors[i])
end
plot(p, legend=:outerbottomright, legend_title="Time [min]", size=(600,400))
savefig(p, "figures/q4_std_ensemble_space.png")


# Stats between ensembles and observations chill
# p = plot(error_stats_storm[!, :Location], error_stats_no_ass[!, :RMSE], label="RMSE", xlabel="Location", ylabel="RMSE", title="RMSE of the ensemble without Data Assimilation")
# savefig(p, "q4_RMSE.png")

# # Amplitude mean
# p= plot(error_stats_storm[!, :Location], error_stats_no_ass[!, :Amplitude_Mean], label="Amplitude Error", xlabel="Location", ylabel="Amplitude Error", title="Amplitude Error of the ensemble without Data Assimilation")
# savefig(p, "q4_Amplitude.png")

# # Timing mean
# p= plot(error_stats_storm[!, :Location], error_stats_no_ass[!, :Timing_Mean], label="Timing Error", xlabel="Location", ylabel="Time [min]", title="Time Error of the ensemble without Data Assimilation")
# savefig(p, "q4_Timing.png")