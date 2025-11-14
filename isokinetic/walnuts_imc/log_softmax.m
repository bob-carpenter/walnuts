function result = log_softmax(X)
    result = X - logsumexp(X);
end