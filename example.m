% This is an example to use AdaTTL algorithm on ‘bibtex’ dataset.
% This example uses LEML as the base large-scale multi-label learning classifier.
%
% Some of the important variables are explained below:
%
% curK: the dimensionality of latent space
% and: the threshold selected by AdaTTL


clc;
clear all;


% load dataset
dataset = 'bibtex';
load(['dataset/', dataset]);

% The sampled percentage of instance
instance_cand = [20, 30, 40, 50]; 
% The percentage of labels to remove
label_cand = [20, 30, 40, 45, 50, 55, 60];
%label_cand = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90];
ni = length(instance_cand);
nl = length(label_cand);

% favorable setting for hyperparameter k
% bibtex: k = 50
% delicious: k = 200
% for larger datasets: k = 500

if (strcmp(dataset, 'bibtex') == 1)
    k = 50;
elseif (strcmp(dataset, 'delicious') == 1)
    k = 200;
else
    k = 500;
end


% 0.02 for wiki10
% 0.2 for eurlex
% 0.6 for delicious
% 0.6 for bibtex
if (strcmp(dataset, 'bibtex') == 1)
	curK = min(k, 0.6 * size(data.Yt, 2));
elseif (strcmp(dataset, 'delicious') == 1)
        curK = min(k, 0.6 * size(data.Yt, 2));
elseif (strcmp(dataset, 'eurlex') == 1)
        curK = min(k, 0.2 * size(data.Yt, 2));
elseif (strcmp(dataset, 'wiki10') == 1)
        curK = min(k, 0.02 * size(data.Yt, 2));
else
        curK = min(k, 0.003 * size(data.Yt, 2));
end


% prec = [P@1, P@3, P@5]
[prec, model_size, test_time] = sampling(data, curK, instance_cand, label_cand);

% calculate reduction ratio
for i = 2: ni * nl
    prec(1, i) = prec(1, i) / prec(1,1);
    prec(2, i) = prec(2, i) / prec(2,1);
    prec(3, i) = prec(3, i) / prec(3,1);
end
prec(1, 1) = 1;
prec(2, 1) = 1;
prec(3, 1) = 1;


% prepare data
x = meshgrid(instance_cand, label_cand);
x = x(:) ./ 100;
y = meshgrid(label_cand, instance_cand)';
y = y(:) ./ 100;

for i = 1: 3
    z = prec(i, :)' ./ max(prec(i, :));
    sf = fit([x, y], z, 'poly24');
    if (i == 1)
        coeff = coeffvalues(sf);
    else
        coeff = coeff + coeffvalues(sf);
    end
end


% calculate the fraction of prediction time reduction
for i = 2: ni * nl
    test_time(i) = test_time(i) / test_time(1);
end
test_time(1) = 1;

z = test_time';
sf = fit([x, y], z, 'poly24');

% alpha is a hyperparameter
% e.g., alpha = 5 for bibtex, alpha = 10 for delicious
alpha = 5;
coeff = coeff - alpha .* coeffvalues(sf);

% calculate the fraction of model size reduction
for i = 2:  ni * nl
    model_size(i) = model_size(i) / model_size(1);
end
model_size(1) = 1;

z = model_size';
% try to fit the sampled datapoints with a polynomial function
sf = fit([x, y], z, 'poly24');
% beta is a hyperparameter
% e.g., beta = 5 for bibtex, beta = 10 for delicious
beta = 5;
coeff = coeff - beta .* coeffvalues(sf);
% trade-off between P@k, testing time, and model size
%


% define the objective
obj = @(y)coeff(1) + coeff(2) + coeff(3)*y + coeff(4) +coeff(5)*y + coeff(6)*y.^2 + coeff(7)*y +coeff(8)*y.^2 + coeff(9)*y.^3 + coeff(10)*y.^2 + coeff(11)*y.^3 + coeff(12)*y.^4;

% get the answer: fraction of tail labels to remove
ans = fminbnd(@(y)-obj(y), 0.1, 0.9)
