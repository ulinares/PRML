using Distributions

sine(x) = sin(2π*x)

function make_sine(N::Int; μ::Float64=0., σ::Float64=0.3)
    d = Normal(μ, σ)
    X = [rand() for _ in 1:N]
    y = [sin(2π*x) + rand(d) for x in X]
    return X, y
end
