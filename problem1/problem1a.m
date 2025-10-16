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


%% LIKELIHOOD 
% compute likelihoods
likelihood_0 = mvnpdf(X, mu0', Sigma0);
likelihood_1 = mvnpdf(X, mu1', Sigma1);

% compute likelihood ratios
likelihood_ratios = likelihood_1 ./ likelihood_0;

%% ROC CURVE

% gamma range
gamma_values = [0, logspace(-3, 6, 1000)];
TPR = zeros(length(gamma_values), 1);  % true positive rate
FPR = zeros(length(gamma_values), 1);  % false Positive rate
FNR = zeros(length(gamma_values), 1);  % false Negative rate

for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    
    % decide based on gamma
    decisions = (likelihood_ratios > gamma);
    
    % calculate performance
    TP = sum(decisions == 1 & labels == 1);     % true positives
    FP = sum(decisions == 1 & labels == 0);     % false positives
    TN = sum(decisions == 0 & labels == 0);     % true negatives
    FN = sum(decisions == 0 & labels == 1);     % false negatives
    
    % calculate rates
    TPR(i) = TP / N1;
    FPR(i) = FP / N0;
    FNR(i) = FN / N1;
end

% add endpoints
TPR = [1; TPR; 0];
FPR = [1; FPR; 0];
FNR = [0; FNR; 1];
gamma_values = [0; gamma_values'; inf];

%% MAXIMUM PROBABILITY OF ERROR

% calculate P(error)
P_error = FPR * p0 + FNR * p1;

% minimum error and corresponding threshold
[min_error, min_idx] = min(P_error);
optimal_gamma_empirical = gamma_values(min_idx);
optimal_TPR = TPR(min_idx);
optimal_FPR = FPR(min_idx);

% theoretical optimal threshold
optimal_gamma_theoretical = p0 / p1;

fprintf('Empirical optimal gamma: %.4f\n', optimal_gamma_empirical);
fprintf('Theoretical optimal gamma: %.4f\n', optimal_gamma_theoretical);
fprintf('Minimum probability of error: %.4f\n', min_error);
fprintf('TPR at minimum P(error): %.4f\n', optimal_TPR);
fprintf('FPR at minimum P(error): %.4f\n', optimal_FPR);

%% PLOT ROC

figure(2);
plot(FPR, TPR, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'g--', 'LineWidth', 1); 
plot(optimal_FPR, optimal_TPR, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - True PDF');
legend('ROC Curve', 'Min P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on; axis square;
xlim([0 1]); ylim([0 1]);