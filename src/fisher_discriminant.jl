mutable struct FisherDiscriminant
    w::Array
    FisherDiscriminant() = new()
end


# Two classes only
function fit!(fd::FisherDiscriminant, X::Array, T::Vector)
    C₁ = X[T .== 1, :]
    C₂ = X[T .== 2, :]
    m₁ = mean(C₁, dims=1)'
    m₂ = mean(C₂, dims=1)'
    S₁ = zeros(2, 2); S₂ = zeros(2, 2)

    for i in 1:size(C₁)[1]
        xₙ = X[i, :]
        S₁ += (xₙ - m₁) * (xₙ - m₁)'
    end

    for i in 1:size(C₂)[1]
        xₙ = X[i, :]
        S₂ += (xₙ - m₂) * (xₙ - m₂)'
    end

    S = S₁ + S₂

    fd.w = S^-1 * (m₂ - m₁)
    nothing
end

function predict(fd::FisherDiscriminant, X::Array)
    [(fd.w' * X[i, :])[1] for i in 1:size(X)[1]]
end

# TODO: Fisher Discriminant for multiple classes
