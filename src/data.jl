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
    X = Matrix{Float64}(undef, n_class * N, 2)
    t = []
    μs = [rand(2) .* 5 .- 2.5 for _ in 1:n_class]
    d = [MvNormal(μ, [σ², σ²]) for μ in μs]

    for i in 1:n_class
        for j in 1:N
            x = rand(d[i])
            X[N*(i-1) + j, :] = x
            push!(t, i)
        end
    end
    return X, t
end


function one_hot(t::Vector)
    classes = length(unique(t))
    N = length(t)
    T = zeros(N, classes)
    for (i, t_) in enumerate(t)
        T[i, t_] = 1
    end
    T
end
