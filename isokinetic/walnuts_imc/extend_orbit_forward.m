function [O, logW] = extend_orbit_forward(logmu, gradlogmu, ...
                                           theta_b, rho_b, W_b, logw_b, ...
                                           h, delta, L, K0)
% Extend orbit forward from boundary state (theta_b, rho_b, W_b, logw_b)
% using IMC-BAB micro-steps and WALNUTS random micro-times.

    O    = {};
    logW = [];

    theta = theta_b;
    rho   = rho_b;
    W     = W_b;
    logw  = logw_b;

    for iter = 1:L
        % forward micro-time selection
        ell_f = micro_imc(logmu, gradlogmu, theta, rho, h, delta, K0);
        ell   = p_micro(ell_f);

        % forward integration with chosen ell
        h_eff = h * 2^(-ell);
        [theta_1, rho_1, dW] = bab_isokinetic(logmu, gradlogmu, theta, rho, h_eff, 2^ell, K0);
        W_1 = W + dW;

        % backward micro-time from the new state (with reversed rho)
        ell_b = micro_imc(logmu, gradlogmu, theta_1, -rho_1, h, delta, K0);

        % weight update in terms of logmu - W
        p_num = pmf_p_micro(ell, ell_b);
        p_den = pmf_p_micro(ell, ell_f);

        if p_num == 0
            logw = -Inf;
        else
            logpi_new = logmu(theta_1) - W_1;
            logpi_old = logmu(theta)   - W;

            logw = logw ...
                 + (logpi_new - logpi_old) ...
                 + log(p_num) - log(p_den);
        end

        % advance boundary state
        theta = theta_1;
        rho   = rho_1;
        W     = W_1;

        O    = [O; {theta, rho, W}];
        logW = [logW; logw];
    end
end