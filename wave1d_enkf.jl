#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
#
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 1 2  3 4  5 6  7  8   # index in state vector ; counting starts at 1
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
#  = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
#  = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

using Interpolations
using Dates
using LinearAlgebra
using DataFrames
using Plots
using Peaks
using Statistics
using Distributions
using StatsPlots
using FileIO, JLD2


struct stats #statistics of peaks
    mean::Float64
    std::Float64
end

plot_maps = false #true or false - plotting makes the runs much slower

minutes_to_seconds = 60.0
hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
seconds_to_hours = 1.0 / hours_to_seconds

function read_series(filename::String)
    infile = open(filename)
    times = DateTime[]
    values = Float64[]
    for line in eachline(infile)
        #println(line)
        if Base.startswith(line, "#") || length(line) <= 1
            continue
        end
        parts = split(line)
        push!(times, DateTime(parts[1], "yyyymmddHHMM"))
        push!(values, parse(Float64, parts[2]))
    end
    close(infile)
    return (times, values)
end

function settings()
    s = Dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s["g"] = 9.81 # acceleration of gravity
    s["D"] = 20.0 # Depth
    s["f"] = 1 / (0.06 * days_to_seconds) # damping time scale
    L = 100.e3 # length of the estuary
    s["L"] = L
    n = 100 #number of cells
    s["n"] = n
    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx = L / (n + 0.5)
    s["dx"] = dx
    x_h = range(0, L - dx, length=n)
    s["x_h"] = x_h
    s["x_u"] = x_h .+ 0.5
    # initial condition
    s["h_0"] = zeros(Float64, n)
    s["u_0"] = zeros(Float64, n)
    # time
    t_f = 2.0 * days_to_seconds #end of simulation
    dt = 10.0 * minutes_to_seconds
    s["dt"] = dt
    reftime = DateTime("201312050000", "yyyymmddHHMM") #times in secs relative
    s["reftime"] = reftime
    t = collect(dt * (1:round(t_f / dt))) #expand to numbers with collect
    s["t"] = t
    #boundary (western water level)
    # read from file
    (bound_times, bound_values) = read_series("tide_cadzand.txt")
    bound_t = zeros(Float64, length(bound_times))
    for i = 1:length(bound_times)
        bound_t[i] = (bound_times[i] - reftime) / Dates.Millisecond(1000)
    end
    s["t_left"] = bound_t
    itp = LinearInterpolation(bound_t, bound_values)
    s["h_left"] = itp(t)

    s["time_cutoff"] = 168 - 39 # [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39] 

    return s
end

function initialize_enfk(s) #return (x,t) at initial time
    #compute initial fields and cache some things for speed
    h_0 = s["h_0"]
    u_0 = s["u_0"]
    n = s["n"]
    x = zeros(2 * n + 1) #order h[0],u[0],...h[n],u[n]
    x[1:2:end-1] = u_0[:]
    x[2:2:end-1] = h_0[:]
    #time
    t = s["t"]
    reftime = s["reftime"]
    dt = s["dt"]
    sec = Dates.Second(1)
    times = []# reftime+sec*t
    for i = 1:length(t)
        push!(times, ((i * dt) * sec) + reftime)
    end
    s["times"] = times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha
    # A and B are tri-diagonal sparse matrices
    Adata_l = zeros(Float64, 2n) #lower diagonal of tridiagonal
    Adata_d = zeros(Float64, 2n + 1)
    Adata_r = zeros(Float64, 2n)
    Bdata_l = zeros(Float64, 2 * n)
    Bdata_d = zeros(Float64, 2 * n + 1)
    Bdata_r = zeros(Float64, 2 * n)

    #left boundary
    Adata_d[1] = 1.0
    #right boundary
    Adata_d[2*n] = 1.0
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
    # = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g = s["g"]
    dx = s["dx"]
    f = s["f"]
    temp1 = 0.5 * g * dt / dx
    temp2 = 0.5 * f * dt
    for i = 2:2:(2*n-1)
        Adata_l[i-1] = -temp1
        Adata_d[i] = 1.0 + temp2
        Adata_r[i] = +temp1
        Bdata_l[i-1] = +temp1
        Bdata_d[i] = 1.0 - temp2
        Bdata_r[i] = -temp1
    end

    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
    # = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D = s["D"]
    temp1 = 0.5 * D * dt / dx
    for i = 3:2:(2*n-1)
        Adata_l[i-1] = -temp1
        Adata_d[i] = 1.0
        Adata_r[i] = +temp1
        Bdata_l[i-1] = +temp1
        Bdata_d[i] = 1.0
        Bdata_r[i] = -temp1
    end

    Adata_d[2*n+1] = 1.0
    Bdata_d[2*n+1] = exp(-1 / 36) # set alpha

    # build sparse matrix
    A = Tridiagonal(Adata_l, Adata_d, Adata_r)
    B = Matrix(Tridiagonal(Bdata_l, Bdata_d, Bdata_r))
    B[1, end] = 1.0
    s["A"] = A #cache for later use
    s["B"] = B

    # Measurement matrix (work out later)
    ilocs = s["ilocs"][2:5]
    H = zeros(Float64, length(ilocs), 2 * n + 1)
    for i = eachindex(ilocs)
        H[i, ilocs[i]] = 1.0
    end
    s["H"] = H

    return (x, t[1])
end

function timestep_enkf(X, t_idx, observations, settings, type)
    # Random noise
    sig_w = 0.2 #0.3224
    w = Normal(0, sig_w^2)
    w_vals = rand(w, size(X, 2))

    # Time update
    A = settings["A"]
    B = settings["B"]
    H = settings["H"]


    R = I * 10e-2
    X[end, :] .+= w_vals # Add noise

    rhs = B * X
    if type ∈ ["new_bc"]
        # rhs[1, :] .+= mean(X[1, :]) #ensemble mean
        # rhs[1,:] .+= X[1, :] #first ensemble member
        rhs[1, :] .+= rand(w, 1) #random noise
    else
        rhs[1, :] .+= settings["h_left"][t_idx] #left boundary
    end

    for i in axes(X, 2)
        X[:, i] = A \ rhs[:, i]
    end

    x_observe = X[settings["ilocs"], 1]

    if type ∈ ["predict"] && t_idx > settings["time_cutoff"]
        return X, x_observe
    end

    # Ensemble average
    x_avg = mean(X, dims=2)

    # Forecast covariance
    P = (X .- x_avg) * (X .- x_avg)' / (size(X, 2) - 1)

    # Measurement update
    K = P * H' * inv(H * P * H' + R)

    if type ∈ ["enkf", "sim_ass", "new_bc", "storm_ass", "predict", "new_ic", "new_ic_vel"]
        X = X + K * (observations .- H * X)
    end

    return X, x_observe
end


function plot_state(x, i, s)
    println("plotting a map.")
    #plot all waterlevels and velocities at one time
    xh = 0.001 * s["x_h"]
    p1 = plot(xh, x[1:2:end], ylabel="h", ylims=(-3.0, 5.0), legend=false)
    xu = 0.001 * s["x_u"]
    p2 = plot(xu, x[2:2:end], ylabel="u", ylims=(-2.0, 3.0), xlabel="x [km]", legend=false)
    p = plot(p1, p2, layout=(2, 1))
    savefig(p, "fig_map_$(string(i,pad=3)).png")
    sleep(0.05) #slow down a bit or the plotting backend starts complaining.
    #This is a bug and will probably be solved soon.
end

function plot_series_enkf(t, X_data, series_data, s, obs_data, type)
    #  X_data = zeros(Float64, length(ilocs), length(t), 50)
    X_data_locs = X_data[s["ilocs"], :, :]

    # plot timeseries from model and observations
    loc_names = s["loc_names"]
    nseries = length(loc_names)
    for i = 1:5
        #fig=PyPlot.figure(i+1)
        std_X = std(X_data_locs[i, :, :], dims=2)
        p = plot(seconds_to_hours .* t, series_data[i, :], ribbon=std_X, linecolor=:blue, label="model")
        ntimes = min(length(t), size(obs_data, 2))
        plot!(p, seconds_to_hours .* t[1:ntimes], obs_data[i, 1:ntimes], linecolor=:black, label="measured")
        title!(p, loc_names[i])
        xlabel!(p, "Time [hours]")
        ylabel!(p, "Water level [m]")
        if type ∈ ["predict"]
            vline!(p, [seconds_to_hours * t[s["time_cutoff"]]], color=:red)
        end
        savefig(p, replace("figures/$(loc_names[i])_$(type).png", " " => "_"))
        sleep(0.05) #Slow down to avoid that that the plotting backend starts complaining. This is a bug and should be fixed soon.
    end
end

function AR_one_step(x_old, sig_w)
    w = Normal(0, sig_w^2)
    w_val = rand(w)

    x_new = exp(-1 / 36) * x_old + w_val

    return x_new
end

function simulate_enkf(n_ensemble, type)
    # for plots
    # locations of observations
    s = settings()
    L = s["L"]
    dx = s["dx"]
    xlocs_waterlevel = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L, 0.99 * L]
    xlocs_velocity = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L]
    ilocs = vcat(map(x -> round(Int, x), xlocs_waterlevel ./ dx) .* 2 .+ 1, map(x -> round(Int, x), xlocs_velocity ./ dx) .* 2 .+ 2)
    println(ilocs)
    loc_names = String[]
    names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    for i = 1:length(xlocs_waterlevel)
        push!(loc_names, "Waterlevel at x=$(0.001*xlocs_waterlevel[i]) km $(names[i])")
    end
    for i = 1:length(xlocs_velocity)
        push!(loc_names, "Velocity at x=$(0.001*xlocs_velocity[i]) km $(names[i])")
    end
    s["xlocs_waterlevel"] = xlocs_waterlevel
    s["xlocs_velocity"] = xlocs_velocity
    s["ilocs"] = ilocs
    s["loc_names"] = loc_names

    if type ∈ ["enkf", "real_ass", "no_ass"]
        #load observations
        (obs_times, obs_values) = read_series("tide_cadzand.txt")
        observed_data = zeros(Float64, length(ilocs), length(obs_times))
        observed_data[1, :] = obs_values[:]
        (obs_times, obs_values) = read_series("tide_vlissingen.txt")
        observed_data[2, :] = obs_values[:]
        (obs_times, obs_values) = read_series("tide_terneuzen.txt")
        observed_data[3, :] = obs_values[:]
        (obs_times, obs_values) = read_series("tide_hansweert.txt")
        observed_data[4, :] = obs_values[:]
        (obs_times, obs_values) = read_series("tide_bath.txt")
        observed_data[5, :] = obs_values[:]

        X_observe = zeros(Float64, length(ilocs), length(obs_times))
    elseif type ∈ ["sim_ass", "new_bc", "new_ic", "new_ic_vel"]
        observed_data = load("data/observed_data_sim.jld2")["observed_data"]
    elseif type ∈ ["storm_ass", "predict"]
        #load observations
        (obs_times, obs_values) = read_series("tide_cadzand.txt")
        observed_data = zeros(Float64, length(ilocs), length(obs_times))
        observed_data[1, :] = obs_values[:]
        (obs_times, obs_values) = read_series("waterlevel_vlissingen.txt")
        observed_data[2, :] = obs_values[:]
        (obs_times, obs_values) = read_series("waterlevel_terneuzen.txt")
        observed_data[3, :] = obs_values[:]
        (obs_times, obs_values) = read_series("waterlevel_hansweert.txt")
        observed_data[4, :] = obs_values[:]
        (obs_times, obs_values) = read_series("waterlevel_bath.txt")
        observed_data[5, :] = obs_values[:]
    end

    (x, t0) = initialize_enfk(s)

    if type ∈ ["new_ic"]
        x_len = length(x)

        x_height = 1:2:x_len-1

        x[x_height] = 2 .* sin.((1:length(x_height)) * 2 * pi / 200)
    elseif type ∈ ["new_ic_vel"]
        x_len = length(x)

        x_height = 1:2:x_len-1

        x[x_height] = 0.5 .* sin.((1:length(x_height)) * 2 * pi / 200)
    end

    # initialize ensemble

    X = zeros(Float64, length(x), n_ensemble)
    for i = 1:n_ensemble
        X[:, i] = x
    end

    t = s["t"]
    times = s["times"]
    series_data = zeros(Float64, length(ilocs), length(t))
    X_data = zeros(Float64, length(x), length(t), n_ensemble)
    nt = length(t)

    for i = 1:nt
        println("timestep $(i), $(round(i/nt*100,digits=1)) %")
        X, x_observe = timestep_enkf(X, i, observed_data[2:5, i], s, type)
        if plot_maps == true
            plot_state(x, i, s) #Show spatial plot. 
            #Very instructive, but turn off for production
        end

        if type ∈ ["no_ass"]
            X_observe[:, i] = x_observe
        end


        series_data[:, i] = mean(X[ilocs, :], dims=2)[:]
        X_data[:, i, :] = X
    end

    if type ∈ ["no_ass"]
        # Save the data for the twin experiment
        save("data/observed_data_sim.jld2", "observed_data", X_observe)
    end


    #plot timeseries
    plot_series_enkf(t, X_data, series_data, s, observed_data, type)

    println("ALl figures have been saved to files.")
    if plot_maps == false
        println("You can plot maps by setting plot_maps to true.")
    else
        println("You can plotting of maps off by setting plot_maps to false.")
        println("This will make the computation much faster.")
    end

    return series_data, observed_data, X_data, s
end

function plot_state_for_gif(X_data, s, observed_data)
    x = mean(X_data, dims=2)[:]
    #plot all waterlevels and velocities at one time
    xh = 0.001 * s["x_h"]
    p1 = plot(xh, X_data[1:2:end-1, :], ylabel="h", ylims=(-3.0, 5.0), xlabel="x [km]", legend=false)
    # errorline!(p1, xh, X_data[1:2:end-1, :], errorstyle=:ribbon, color=:blue)
    scatter!(p1, s["xlocs_waterlevel"] ./ 1000, observed_data[1:5])

    xu = 0.001 * s["x_u"]
    p2 = plot(xu, X_data[2:2:end-1, :], ylabel="u", ylims=(-2.0, 3.0), xlabel="x [km]", legend=false)
    # errorline!(p2, xu, X_data[2:2:end], errorstyle=:ribbon, color=:blue)
    p = plot(p1, p2, layout=(2, 1))

    return p
end


types = ["enkf", "no_ass", "sim_ass", "new_bc", "storm_ass", "predict", "new_ic"]


for n_ensemble ∈ [50]#[5, 10, 20, 50, 100]
    type = "new_ic_vel"
    series_data, observed_data, X_data, s = simulate_enkf(n_ensemble, type)

    @save "data/X_data_$(type).jld2" X_data #_$(168-s["time_cutoff"])

    anim = @animate for i ∈ 1:length(s["t"])
        plot_state_for_gif(X_data[:, i, :], s, observed_data[:, i+1])
    end


    gif(anim, "figures/animation_$(type).gif", fps=15)
end

# type = "sim_ass"

# series_data, observed_data, X_data, s = simulate_enkf(50, type)

# @save "data/X_data_$(type).jld2" X_data


# anim = @animate for i ∈ 1:length(s["t"])
#     plot_state_for_gif(X_data[:, i, :], s, observed_data[:, i])
# end


# gif(anim, "figures/animation_$(type).gif", fps=15)
