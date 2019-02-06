module TempBay
using Distributions
using LinearAlgebra

#mutable struct BayesianRegression{S <: Array, T <: Array}
#    mₙ::S
#    Sₙ::T
#    BayesianRegression{S, T}() where {S, T} = new()
#end

mutable struct BayesianRegression
    mₙ::Array
    Sₙ::Array
    BayesianRegression() = new()
end

function prior(bay_reg::BayesianRegression, α::Float64)
    m = length(bay_reg.coefs)
    d = MvNormal(zeros(m), ones(m) / α)
    return d
end

# Posterior distribution parameters
# Valid only for isotropic gaussian prior
function Sₙ(Φ::Matrix, α::Real, β::Real)
    m = size(Φ)[2]
    Id = Matrix{Float64}(I, m, m)
    S_inverse = α .* Id + β .* Φ' * Φ
    return S_inverse^-1
end

function mₙ(Φ::Matrix, y::Array, Sₙ::Matrix, β::Float64)
    y = reshape(y, :, 1)
    return β .* Sₙ * Φ' * y
end

function fit!(bay_reg::BayesianRegression, Φ::Array, y::Array;
    α::Float64=1., β::Float64=1/0.3^2)
    bay_reg.Sₙ = Sₙ(Φ, α, β)
    bay_reg.mₙ = mₙ(Φ, y, bay_reg.Sₙ, β)
    nothing
end



end
