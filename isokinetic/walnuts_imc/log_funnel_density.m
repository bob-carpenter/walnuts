function log_p = log_funnel_density(theta)
    v = theta(1);
    x = theta(2:end);
    log_p_v = -0.5 * (v^2 / 9);
    log_p_x = -0.5*sum(x.^2)/exp(v)-v*length(x)/2;
    log_p = log_p_v + log_p_x;
end