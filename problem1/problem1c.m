% Author: Anthony Yalong
% NUID: 002156860

%% DATA GENERATION
clear all; close all;

% generate data
N = 10000;
p0 = 0.65;
p1 = 0.35;

u = rand(1,N) >= p0;
N0 = length(find(u == 0));
N1 = length(find(u == 1));

% generate class 0 samples
mu0 = [-1/2; -1/2; -1/2];
Sigma0 = [1, -0.5, 0.3; -0.5, 1, -0.5; 0.3, -0.5, 1];
r0 = mvnrnd(mu0', Sigma0, N0);

% visualize class 0
figure(1);
plot3(r0(:,1), r0(:,2), r0(:,3), '.b');
hold on; axis equal;

% generate class 1 samples
mu1 = [1; 1; 1];
Sigma1 = [1, 0.3, -0.2; 0.3, 1, 0.3; -0.2, 0.3, 1];
r1 = mvnrnd(mu1', Sigma1, N1);

% visualize class 1
figure(1);
plot3(r1(:,1), r1(:,2), r1(:,3), '.r');
hold on; axis equal;

% combine data and labels
X = [r0; r1];
labels = [zeros(N0,1); ones(N1,1)];


%% PARAMETER ESTIMATION
% setup
X0 = X(labels == 0, :);
X1 = X(labels == 1, :);

% estimate parameters
m0_est = mean(X0)';  % class 0 mean
m1_est = mean(X1)';  % class 1 mean
C0_est = cov(X0);    % class 0 covariance
C1_est = cov(X1);    % class 1 covariance

%% LDA PROJECTION
% between-class scatter
Sb = (m1_est - m0_est) * (m1_est - m0_est)';

% within-class scatter
Sw = 0.5 * C0_est + 0.5 * C1_est;

% projection vector
w_LDA = Sw \ (m1_est - m0_est);
w_LDA = w_LDA / norm(w_LDA);  % normalize

% project data
projections = X * w_LDA;

%% ROC CURVE
tau_values = linspace(min(projections)-1, max(projections)+1, 1000);
TPR = zeros(length(tau_values), 1);  % true positive rate
FPR = zeros(length(tau_values), 1);  % false positive rate
FNR = zeros(length(tau_values), 1);  % false negative rate

for i = 1:length(tau_values)
    tau = tau_values(i);
    
    % decide based on projection threshold
    decisions = (projections > tau);
    
    % calculate performance
    TP = sum(decisions == 1 & labels == 1);  % true positives
    FP = sum(decisions == 1 & labels == 0);  % false positives
    FN = sum(decisions == 0 & labels == 1);  % false negatives
    
    % calculate rates
    TPR(i) = TP / N1;
    FPR(i) = FP / N0;
    FNR(i) = FN / N1;
end

%% MAXIMUM PROBABILITY OF ERROR
% calculate P(error)
P_error = FPR * p0 + FNR * p1;

% minimum error and corresponding threshold
[min_error, min_idx] = min(P_error);
optimal_tau = tau_values(min_idx);
optimal_TPR = TPR(min_idx);
optimal_FPR = FPR(min_idx);

fprintf('Optimal threshold tau: %.4f\n', optimal_tau);
fprintf('Minimum probability of error: %.4f\n', min_error);
fprintf('TPR at minimum error: %.4f\n', optimal_TPR);
fprintf('FPR at minimum error: %.4f\n', optimal_FPR);

%% PLOT ROC
figure(2);
plot(FPR, TPR, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'g--', 'LineWidth', 1);
plot(optimal_FPR, optimal_TPR, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Fisher LDA');
legend('ROC Curve', 'Min P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on; axis square;
xlim([0 1]); ylim([0 1]);

%% VISUALIZE PROJECTIONS
figure(3);
edges = linspace(min(projections), max(projections), 50);
histogram(projections(labels==0), edges, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(projections(labels==1), edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xline(optimal_tau, 'g--', 'LineWidth', 2);
xlabel('Projected Value');
ylabel('Count');
title('1D Projections');
legend('Class 0', 'Class 1', 'Optimal Threshold');