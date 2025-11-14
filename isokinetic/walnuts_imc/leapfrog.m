function [theta, rho] = leapfrog(logmu, gradlogmu, theta, rho, h, ell)
    for k = 1:ell
        rho_half = rho + (h / 2) * gradlogmu(theta);
        theta = theta + h * rho_half;
        rho = rho_half + (h / 2) * gradlogmu(theta);
    end
end
