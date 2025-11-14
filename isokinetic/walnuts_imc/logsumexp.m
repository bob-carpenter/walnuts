function result = logsumexp(X)
    maxX = max(X);
    result = maxX + log(sum(exp(X - maxX)));
end