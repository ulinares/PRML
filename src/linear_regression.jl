using LinearAlgebra
# A simple linear regression

# A structure to holds the model parameters

mutable struct LinearRegression{T <: Array}
    λ::Float64
    coefs::T
    σ::Float64
    LinearRegression{T}(λ) where T = new{T}(λ)
end


LinearRegression(; eltype=Float64, λ=0.) = LinearRegression{Array{eltype, 2}}(λ)

function fit!(lm::LinearRegression, Φ::S, y::T) where {S, T}
    if lm.λ == 0
        N, m = size(Φ)
        w_ml = (Φ' * Φ)^-1 * Φ'* y
        lm.coefs = Array{Float64, 2}(w_ml)
        lm.σ = sqrt(sum([(y[i] - (lm.coefs' * Φ[i, :])[1])^2 for i in 1:N]) / N)
    else
        id_size = size(Φ)[2]
        I = Matrix{Float64}(I, id_size, id_size)
        w_ml = (λ .* I + Φ' * Φ)^-1 + Φ'*y
        lm.coefs = Array{Float64, 1}([w_ml])
    end
end

# TODO: implement online learning via the partial_fit function
function partial_fit!(lm::LinearRegression)
end

function predict(lm::LinearRegression, X::T) where T <: Array
    y_preds = Float64[]
    for x in X
        y = lm.coefs * x
        push!(y_preds, y)
    end
    return y_preds
end
