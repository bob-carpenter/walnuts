clear all;

n_samples=1e5;
d = 11;
theta = zeros(d, 1);
h = 0.5;
i_max = 10;
amin = 0.70;
delta = log(1 / amin);

res = cell(n_samples, 1); % Use cell array for variable-length storage
T_vec = zeros(n_samples, 1);

tic
for i = 1:n_samples
    [theta, T] = walnuts(@log_funnel_density, @grad_log_funnel_density, theta, h, i_max, delta);
    res{i} = theta;
    T_vec(i) = T;
end
toc

X = zeros(n_samples, 2);
for i = 1:n_samples
    X(i, :) = res{i}(1:2)'; % Extract first two components and transpose to row
end

%%
% Generate grid for contour plot
x_range = linspace(-15, 15, 100);
y_range = linspace(-200, 200, 100);
[X_grid, Y_grid] = meshgrid(x_range, y_range);
Z_grid = zeros(size(X_grid));

% Compute log density for each grid point
for i = 1:size(X_grid, 1)
    for j = 1:size(X_grid, 2)
        theta = [X_grid(i, j); Y_grid(i, j); zeros(d-2, 1)]; % Extend to 11D
        Z_grid(i, j) = log_funnel_density(theta);
    end
end

% Normalize to avoid large negative values (optional)
%Z_grid = exp(Z_grid - max(Z_grid(:))); % Convert log-density to probability scale


hstr=num2str(h,3);
Num = strfind(hstr,'.');
hstr(Num)='p';

imaxstr=num2str(i_max,3);
Num = strfind(imaxstr,'.');
imaxstr(Num)='p';

aminstr=num2str(amin,3);
Num = strfind(aminstr,'.');
aminstr(Num)='p';


% scatter plot
figure(1); 
clf; hold on;
contour(X_grid, Y_grid, Z_grid, [-48 -30 -4], 'LineWidth', 2); % Contour plot of density
scatter(X(:,1), X(:,2), 25, 'filled', 'k', 'MarkerFaceAlpha', 0.01); % Sampled points
%xlabel('Component 1');
%ylabel('Component 2');
%title('Scatter plot of first two components of res with funnel density contours');
xlim([-15 15]); ylim([-200 200])
grid on;
box on; 
set(gcf,'color',[1.0,1.0,1.0]);
set(gca,'FontSize',20);


%filename=['funnel_scatter_h_' hstr '_amin_' aminstr '.pdf']; box on;
%exportgraphics(gca,filename,'Resolution',150)


% histogram density
figure(2); clf; hold on;
v_vec=X(:,1);
[n walnuts_em vout]=kde(v_vec(:),2^14,-14,14);
exact_im=exp(-0.5*(vout.^2)/9.0);
walnuts_em=walnuts_em/sum(walnuts_em(:));
exact_im=exact_im/sum(exact_im(:));
plot(vout,walnuts_em,'k','LineWidth',2);
plot(vout,exact_im,'k','LineWidth',2,'color',[0.75 0.75 0.75]);
xlim([-12 12]);
xlabel('$\omega$','FontSize',16,'Interpreter','latex');
box on;
grid on;
set(gcf,'color',[1.0,1.0,1.0]);
legend({'walnuts', 'exact'}, 'location', 'northeast', 'Interpreter','latex', 'fontsize',20, 'Orientation','vertical');


%filename=['funnel_density_h_' hstr '_amin_' aminstr '.pdf']; box on;
%exportgraphics(gca,filename,'Resolution',150)
