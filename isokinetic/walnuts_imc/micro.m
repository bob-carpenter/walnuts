function ell = micro(logmu, gradlogmu, theta_0, rho_0, h_0, delta)
    ell = 0;

    while true
        theta = theta_0;
        rho = rho_0;
        R=2^(ell);
        h = h_0/R;
        H_max = -logmu(theta) + 0.5 * norm(rho)^2;
        H_min = H_max;
        
        for j = 1:R
            rho_half = rho + (h / 2) * gradlogmu(theta);
            theta = theta + h * rho_half;
            rho = rho_half + (h / 2) * gradlogmu(theta);
            H = -logmu(theta) + 0.5 * norm(rho)^2;
            H_max = max(H, H_max);
            H_min = min(H, H_min);
        end
        
        if H_max - H_min <= delta
            return;
        end
        ell = ell + 1;
    end
end