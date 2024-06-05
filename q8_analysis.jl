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
    peak_model_idcs, peak_model_vals = keep_positive(peak_model_idcs, peak_model_vals)
    (peak_obs_idcs, peak_obs_vals, _) = findmaxima(observed_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)
    peak_obs_idcs, peak_obs_vals = keep_positive(peak_obs_idcs, peak_obs_vals)

    (trough_model_idcs, trough_model_vals, _) = findminima(model_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)
    trough_model_idcs, trough_model_vals = keep_negative(trough_model_idcs, trough_model_vals)
    (trough_obs_idcs, trough_obs_vals, _) = findminima(observed_data) |> peakproms!() |> peakwidths!(; min=peak_min_tolerance)
    trough_obs_idcs, trough_obs_vals = keep_negative(trough_obs_idcs, trough_obs_vals)

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
observed_data = load("data/observed_data.jld2")["observed_data"]
X_data = load("data/X_data_new_bc_50.jld2")["X_data"]
ilocs = [1, 51, 101, 151, 199]

ensemble_mean = mean(X_data[ilocs,:,:], dims=3)


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
    ensemble_mean_smooth = filtfilt(ones(10)/10, ensemble_mean[i, :, 1])
    observed_data_smooth = filtfilt(ones(10)/10, observed_data[i, 2:end])

    amplitude_error, timing_error = peak_statistic(ensemble_mean_smooth, observed_data_smooth)
    rmse, bias = compute_rmse_bias(ensemble_mean_smooth, observed_data_smooth)

    error_stats[i, :RMSE] = round(rmse, digits=2)
    error_stats[i, :Bias] = round(bias, digits=2)
    error_stats[i, :Amplitude_Mean] = round(amplitude_error.mean, digits=2)
    error_stats[i, :Amplitude_Std] = round(amplitude_error.std, digits=2)
    error_stats[i, :Timing_Mean] = round(timing_error.mean, digits=2)
    error_stats[i, :Timing_Std] = round(timing_error.std, digits=2)
end

latexify(error_stats, env=:table, latex=false)
