function grad_log_p = grad_log_funnel_density(theta)
    v = theta(1);
    x = theta(2:end);
    grad_log_p_v = -v/9+0.5*sum(x.^2)/exp(v)-length(x)/2;
    grad_log_p_x = -x/exp(v);
    grad_log_p = [grad_log_p_v; grad_log_p_x(:)];
end