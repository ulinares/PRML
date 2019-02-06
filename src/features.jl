basis_expansion(Φ, X, f) = hcat(Φ, f.(X))

σ(x) = 1 / (1 + exp(-x))
S(x, μ, s) = σ((x - μ) / s)
G(x, μ, s) = exp(-(x - μ)^2 / (2*s^2))


function poly_expansion(X; degree::Int=2, bias::Bool=true)
    if bias
        Φ = ones(length(X), 1)
        Φ = basis_expansion(Φ, X, x -> x)
    else
        Φ = reshape(X, :, 1)
    end
    for i in 2:degree
        Φ = basis_expansion(Φ, X, x -> x^i)
    end 
    Φ
end

function sigmoidal_expansion(X; μs::Union{Vector, StepRangeLen},
    s::Float64, bias::Bool=true)
    if bias
        Φ = ones(length(X), 1)
        Φ = basis_expansion(Φ, X, x -> S(x, μs[1], s))
    else
        X_ = reshape(X, :, 1)
        Φ = S.(X_, μs[1], s)
    end
    for μ in μs[2:end]
        Φ = basis_expansion(Φ, X, x -> S(x, μ, s))
    end
    Φ
end

function gaussian_expansion(X; μs::Union{Vector, StepRangeLen},
    s::Float64, bias::Bool=true)
    if bias
        Φ = ones(length(X), 1)
        Φ = basis_expansion(Φ, X, x -> G(x, μs[1], s))
    else
        X_ = reshape(X, :, 1)
        Φ = G.(X_, μs[1], s)
    end
    for μ in μs[2:end]
        Φ = basis_expansion(Φ, X, x -> G(x, μ, s))
    end
    Φ
end
