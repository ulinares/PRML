module Temp
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
    y = reshape(y, :, 1)
    N, m = size(Φ)
    if lm.λ == 0
        w_ml = (Φ' * Φ)^-1 * Φ'* y
        lm.coefs = Array{Float64, 2}(w_ml)
    else
        λ = lm.λ
        Id = Matrix{Float64}(I, m, m)
        w_ml = (λ .* Id + Φ' * Φ)^-1 * Φ'*y
        lm.coefs = Array{Float64, 2}(w_ml)
    end
    lm.σ = sqrt(sum([(y[i] - (lm.coefs' * Φ[i, :])[1])^2 for i in 1:N]) / N)
    nothing
end

# TODO: implement online learning via the partial_fit function
function partial_fit!(lm::LinearRegression)
end

function predict(lm::LinearRegression, X::T) where T <: Array
    y_preds = []
    for i in 1:size(X)[1]
        y = dot(X[i:i, :], lm.coefs)
        push!(y_preds, y)
    end
    return y_preds
end

end
