using Distributions

abstract type Data end

struct SynData <: Data
    h::Function
    σ::Float64
    μ::Float64
end

SynData(h::Function, σ::Float64) = SynData(h, σ, 0.)

function make_data(d::Data, n::Int; X=false)
    d = Normal(d.μ, d.σ)
    if !X
        X = rand(n)
    end
    y = [d.h(x) + rand(d) for x in X]
    [X, y]
end
