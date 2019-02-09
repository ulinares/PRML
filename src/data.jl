using Distributions

sine(x) = sin(2π*x)

function make_sine(N::Int; μ::Float64=0., σ::Float64=0.3)
    d = Normal(μ, σ)
    X = [rand() for _ in 1:N]
    y = [sin(2π*x) + rand(d) for x in X]
    return X, y
end

# Generate data by clases with same covariance matrix
# TODO: generate with different covariance?
function make_classes(N::Int; n_class::Int=2, σ²=0.25)
    X = []
    T = []
    μs = [rand(2) .* 5 .- 2.5 for _ in 1:n_class]
    d = [MvNormal(μ, [σ², σ²]) for μ in μs]
    for i in 1:n_class
        for _ in 1:N
            x = rand(d[i])
            t = zeros(n_class); t[i] = 1
            push!(X, x)
            push!(T, t)
        end
    end
    return X, T
end
