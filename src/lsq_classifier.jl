module Classifier
mutable struct LsqClassifier
    W̃::Matrix
    LsqClassifier() = new()
end

function fit!(lsq_clf::LsqClassifier, X̃::Matrix, T::Matrix)
    lsq_clf.W̃ = (X̃' * X̃)^-1 * X̃' * T
    nothing
end

function fit!(lsq_clf::LsqClassifier, X̃::Matrix, T::Vector)
    fit!(lsq_clf, X̃, T)
    nothing
end

function predict(lsq_clf::LsqClassifier, X̃::Matrix)
    preds = [argmax(lsq_clf.W̃' * X[i, :]) for i in 1:size(X̃)[1]]
    preds
end

# TODO: add plot_decision function
function plot_decision()

end
end
