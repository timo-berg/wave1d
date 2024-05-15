using Distributions, Random
using Statistics

function AR_one(n, sig_w)
    x = zeros(n)
    w = Normal(0, sig_w)
    w_val = rand(w, n)

    for t in 2:n
        x[t] = exp(-1 / 36) * x[t-1] + w_val[t]
    end

    return x
end



println(std(AR_one(10000000, 0.0115)))
# gives 0.1999462842187366
plot(std(AR_one(10000000, 0.3224)))