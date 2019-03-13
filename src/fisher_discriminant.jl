mutable struct FisherDiscriminant
    w::Array
    _m1::Array
    _m2::Array
    _S1::Array
    _S2::Array
    _S::Array
    FisherDiscriminant() = new()
end


# Two classes only
function fit!(fd::FisherDiscriminant, X::Array, T::Vector)
    C₁ = X[T .== 1, :]
    C₂ = X[T .== 2, :]
    fd._m1 = mean(C₁, dims=1)'
    fd._m2 = mean(C₂, dims=1)'
    fd._S1 = zeros(2, 2); fd._S2 = zeros(2, 2)

    for i in 1:size(C₁)[1]
        xₙ = X[i, :]
        fd._S1 += (xₙ - fd._m1) * (xₙ - fd._m1)'
    end

    for i in 1:size(C₂)[1]
        xₙ = X[i, :]
        fd._S2 += (xₙ - fd._m2) * (xₙ - fd._m2)'
    end

    fd._S = fd._S1 + fd._S2

    fd.w = fd._S^-1 * (fd._m2 - fd._m1)
    nothing
end

function predict(fd::FisherDiscriminant, X::Array)
    [(fd.w' * X[i, :])[1] for i in 1:size(X)[1]]
end

# TODO: Fisher Discriminant for multiple classes
