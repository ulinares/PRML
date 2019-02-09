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
function make_classes(N::Int; n_class::Int=2, σ²=0.25,
    one_hot::Bool=true)
    X = []
    t = []
    T = zeros(n_class * N, n_class)
    μs = [rand(2) .* 5 .- 2.5 for _ in 1:n_class]
    d = [MvNormal(μ, [σ², σ²]) for μ in μs]

    for i in 1:n_class
        for _ in 1:N
            x = rand(d[i])
            push!(X, x)
            push!(t, i)
        end
    end

    if one_hot
        for (i, class) in enumerate(t)
            T[i, class] = 1.
        end
        return X, T
    else
        return X, t
    end
end
