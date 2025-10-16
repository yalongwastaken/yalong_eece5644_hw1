% Author: Anthony Yalong
% NUID: 002156860

%% SETUP
clear all; close all;

% random seed
rng(42);

% class setup
priors = [0.25; 0.25; 0.25; 0.25];
K = 4;

% class means
mu = cell(K, 1);
mu{1} = [-3; 0];
mu{2} = [3; 0];
mu{3} = [0; 3];
mu{4} = [0; 1];

% covariance matrices
Sigma = cell(K, 1);
Sigma{1} = [1.5, 0.3; 0.3, 1];
Sigma{2} = [1, -0.3; -0.3, 1.5];
Sigma{3} = [1.2, 0; 0, 1.2]; 
Sigma{4} = [0.8, 0.2; 0.2, 0.8];

%% GENERATE DATA
N = 10000;
d = length(mu{1});
X = zeros(N, d);
labels = zeros(N, 1);

for i = 1:N
    u = rand();
    if u < priors(1)
        class = 1;
    elseif u < sum(priors(1:2))
        class = 2;
    elseif u < sum(priors(1:3))
        class = 3;
    else
        class = 4;
    end
    
    % generate sample from selected class
    X(i, :) = mvnrnd(mu{class}', Sigma{class});
    labels(i) = class;
end

% count samples per class
for k = 1:K
    Nk = sum(labels == k);
    fprintf('Class %d: %d samples (%.1f%% vs expected 25%%)\n', ...
            k, Nk, 100*Nk/N);
end

%% MAP CLASSIFIER

% compute likelihood
likelihoods = zeros(N, K);
for k = 1:K
    likelihoods(:, k) = mvnpdf(X, mu{k}', Sigma{k});
end

% unnormalized posteriors
posteriors = likelihoods .* repmat(priors', N, 1);

% MAP decision
[~, decisions_MAP] = max(posteriors, [], 2);

%% CONFUSION MATRIX

confusion_MAP = zeros(K, K);
for true_class = 1:K
    for decided_class = 1:K
        count = sum(decisions_MAP == decided_class & labels == true_class);
        confusion_MAP(decided_class, true_class) = count / sum(labels == true_class);
    end
end

fprintf('Confusion Matrix (rows=decisions, cols=true labels):\n');
fprintf('        L=1     L=2     L=3     L=4\n');
for d = 1:K
    fprintf('D=%d   ', d);
    for l = 1:K
        fprintf('%.4f  ', confusion_MAP(d, l));
    end
    fprintf('\n');
end

% overall error rate
P_error_MAP = 1 - trace(confusion_MAP) / K;  % Average of diagonal
fprintf('\nOverall P(error) = %.4f (%.2f%%)\n', P_error_MAP, P_error_MAP*100);

% class-specific errors
fprintf('\nClass-specific accuracy:\n');
for k = 1:K
    accuracy = confusion_MAP(k, k);
    fprintf('Class %d: %.2f%% correct\n', k, accuracy*100);
end

%% VISUALIZATION
figure();

% markers
markers = {'o', '^', 's', 'd'};

% green shades
greens = [0.0 0.5 0.0;
          0.0 0.7 0.0;
          0.0 0.9 0.0;
          0.4 1.0 0.4];

% red shade
red = [0.8 0.0 0.0];

% plot each class
for k = 1:K
    idx_class = (labels == k);
    
    % correct
    idx_correct = idx_class & (decisions_MAP == k);
    if any(idx_correct)
        scatter(X(idx_correct,1), X(idx_correct,2), 30, greens(k,:), markers{k}, 'filled');
        hold on;
    end
    
    % incorrect
    idx_wrong = idx_class & (decisions_MAP ~= k);
    if any(idx_wrong)
        scatter(X(idx_wrong,1), X(idx_wrong,2), 30, red, markers{k}, 'filled');
        hold on;
    end
end

% plot info
xlabel('Feature 1');
ylabel('Feature 2');
title(sprintf('MAP Classification (Error = %.2f%%)', P_error_MAP*100));
grid on;
axis equal;

%  legend
h1 = scatter(NaN, NaN, 50, greens(1,:), 'o', 'filled');
h2 = scatter(NaN, NaN, 50, greens(2,:), '^', 'filled');
h3 = scatter(NaN, NaN, 50, greens(3,:), 's', 'filled');
h4 = scatter(NaN, NaN, 50, greens(4,:), 'd', 'filled');
legend([h1 h2 h3 h4], {'Class 1', 'Class 2', 'Class 3', 'Class 4'}, 'Location', 'bestoutside');