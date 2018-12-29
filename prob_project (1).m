data1 = load('data.mat')
clc
F1 = data1.F1
F2 = data1.F2

% Form the train and test data
F1_train = F1(1:100,:);
F2_train = F2(1:100,:);
F1_test = F1(101:1000,:);
F2_test = F2(101:1000,:);

% Define mean and standeviation matrices
means_F1 = zeros(1,5);
std_dev_F1 = zeros(1, 5);
means_F2 = zeros(1,5);
std_dev_F2 = zeros(1, 5);
means_Z1 = zeros(1,5);
std_dev_Z1 = zeros(1, 5);

% Define Normalised Matrices
F1_normal = zeros(900, 5);
F2_normal = zeros(900, 5);
Z1 = zeros(900, 5);
Z1_normal = zeros(900, 5);
Z1_normal_probs = zeros(900, 5);
F2_normal_probs = zeros(900, 5);
Z1F2_normal = zeros(900, 5);
Z1F1_normal_probs = zeros(900, 5);


% Define index matrix 
F1_normal_index = zeros(900, 5);
F2_normal_index = zeros(900, 5);
original_index = zeros(900, 5);

probs = zeros(1,5);
probs2 = zeros(1,5);
probs_Z1 = zeros(1,5);

% Using training set to calculate the mean and standard deviation
for i = 1:5
   means_F1(i) = mean(F1_train(:,i));
   std_dev_F1(i) = std(F1_train(:,i));
end

% Getting indexes of argsmax in Original Matrix
for i = 1:5
   original_index(:, i) = i;
end


% F1
% Finding Probabilities from Z scores thus calculated
for j = 1:900
    for i = 1:5
        vals = F1_test(j,i);
        probs(1) = normpdf(vals, means_F1(1), std_dev_F1(1));
        probs(2) = normpdf(vals, means_F1(2), std_dev_F1(2));
        probs(3) = normpdf(vals, means_F1(3), std_dev_F1(3));
        probs(4) = normpdf(vals, means_F1(4), std_dev_F1(4));
        probs(5) = normpdf(vals, means_F1(5), std_dev_F1(5));
        [val, loc] = max(probs);
        F1_normal(j, i) = loc;
    end
end

%Checking Classification accuracy and Error Rate
corr_preds_F1 = zeros(1,5);
incorr_preds_F1 = zeros(1,5);
for i = 1:5
    counts = 0;
    misclass = 0;
    for j = 1:900
        if (F1_normal(j,i) - original_index(j,i) == 0)
            counts = counts + 1;
        else
            misclass = misclass + 1;
        end
    end
    corr_preds_F1(i) = counts;
    incorr_preds_F1(i) = misclass;
end

classify_acc = zeros(1,5);
Error_rates = zeros(1,5);
for i = 1:5
    classify_acc(i) = (corr_preds_F1(i)/900)*100;
    Error_rates(i) = (incorr_preds_F1(i)/900)*100;
end

diff_mat = F1_normal - original_index;
error_vals = diff_mat==0;
classification_accuracy = (sum(error_vals(:))/4500)*100;
error_rate = 100-classification_accuracy;

% Z1

for i = 1:1000
Z1(i,:) = zscore(F1(i,:));
end

hold on
for i = 1:5
scatter(Z1(:,i),F2(:,i))
end
title('Scatterplot: Normalized Features')
xlabel('1^{st} Feature (Z1)')
ylabel('2^{nd} Feature (F2)')
legend('C1','C2','C3','C4','C5')
hold off

Z1_train = Z1(1:100,:);
Z1_test = Z1(101:1000,:);

for i = 1:5
   means_Z1(i) = mean(Z1_train(:,i));
   std_dev_Z1(i) = std(Z1_train(:,i));
end

% Finding Probabilities from Z scores thus calculated
for j = 1:900
    for i = 1:5
        vals = Z1_test(j,i);
        probs_Z1(1) = normpdf(vals, means_Z1(1), std_dev_Z1(1));
        probs_Z1(2) = normpdf(vals, means_Z1(2), std_dev_Z1(2));
        probs_Z1(3) = normpdf(vals, means_Z1(3), std_dev_Z1(3));
        probs_Z1(4) = normpdf(vals, means_Z1(4), std_dev_Z1(4));
        probs_Z1(5) = normpdf(vals, means_Z1(5), std_dev_Z1(5));
        
        [val, loc] = max(probs_Z1);
        Z1_normal(j, i) = loc;
    end
end

%Checking Classification accuracy and Error Rate
corr_preds_Z1 = zeros(1,5);
incorr_preds_Z1 = zeros(1,5);
for i = 1:5
    counts = 0;
    misclass = 0;
    for j = 1:900
        if (Z1_normal(j,i) - original_index(j,i) == 0)
            counts = counts + 1;
        else
            misclass = misclass + 1;
        end
    end
    corr_preds_Z1(i) = counts;
    incorr_preds_Z1(i) = misclass;
end

classify_accZ1 = zeros(1,5);
Error_ratesZ1 = zeros(1,5);
for i = 1:5
    classify_accZ1(i) = (corr_preds_Z1(i)/900)*100;
    Error_ratesZ1(i) = (incorr_preds_Z1(i)/900)*100;
end

diff_matZ1 = Z1_normal - original_index;
error_valsZ1 = diff_matZ1==0;
classification_accuracyZ1 = (sum(error_valsZ1(:))/4500)*100;
error_rateZ1 = 100-classification_accuracyZ1;

% F2
for i = 1:5
   means_F2(i) = mean(F2_train(:,i));
   std_dev_F2(i) = std(F2_train(:,i));
end

for j = 1:900
    for i = 1:5
        vals2 = F2_test(j,i);
        probs2(1) = normpdf(vals2, means_F2(1), std_dev_F2(1));
        probs2(2) = normpdf(vals2, means_F2(2), std_dev_F2(2));
        probs2(3) = normpdf(vals2, means_F2(3), std_dev_F2(3));
        probs2(4) = normpdf(vals2, means_F2(4), std_dev_F2(4));
        probs2(5) = normpdf(vals2, means_F2(5), std_dev_F2(5));
        [val1, loc] = max(probs2);
        F2_normal(j, i) = loc;
    end
end

corr_preds_F2 = zeros(1,5);
incorr_preds_F2 = zeros(1,5);
for i = 1:5
    counts = 0;
    misclass = 0;
    for j = 1:900
        if (F2_normal(j,i) - original_index(j,i) == 0)
            counts = counts + 1;
        else
            misclass = misclass + 1;
        end
    end
    corr_preds_F2(i) = counts;
    incorr_preds_F2(i) = misclass;
end

classify_acc2 = zeros(1,5);
Error_rates2 = zeros(1,5);
for i = 1:5
    classify_acc2(i) = (corr_preds_F2(i)/900)*100;
    Error_rates2(i) = (incorr_preds_F2(i)/900)*100;
end

diff_mat2 = F2_normal - original_index;
error_vals2 = diff_mat2==0;
classification_accuracy2 = (sum(error_vals2(:))/4500)*100;
error_rate2 = 100-classification_accuracy2;

% Multipy Z1 and F2
probsZ1F2 = zeros(1, 5);
% Finding Probabilities from Z scores thus calculated
for j = 1:900
    for i = 1:5
        vals = F2_test(j,i);
        probs_F2(1) = normpdf(vals, means_F2(1), std_dev_F2(1));
        probs_F2(2) = normpdf(vals, means_F2(2), std_dev_F2(2));
        probs_F2(3) = normpdf(vals, means_F2(3), std_dev_F2(3));
        probs_F2(4) = normpdf(vals, means_F2(4), std_dev_F2(4));
        probs_F2(5) = normpdf(vals, means_F2(5), std_dev_F2(5));
        vals = Z1_test(j,i);
        probs_Z1(1) = normpdf(vals, means_Z1(1), std_dev_Z1(1));
        probs_Z1(2) = normpdf(vals, means_Z1(2), std_dev_Z1(2));
        probs_Z1(3) = normpdf(vals, means_Z1(3), std_dev_Z1(3));
        probs_Z1(4) = normpdf(vals, means_Z1(4), std_dev_Z1(4));
        probs_Z1(5) = normpdf(vals, means_Z1(5), std_dev_Z1(5));
        
        probsZ1F2 = probs_F2 .* probs_Z1;
        
        [val, loc] = max(probsZ1F2);
        Z1F2_normal(j, i) = loc;
    end
end

corr_preds_Z1F2 = zeros(1,5);
incorr_preds_Z1F2 = zeros(1,5);
for i = 1:5
    counts = 0;
    misclass = 0;
    for j = 1:900
        if (F2_normal(j,i) - original_index(j,i) == 0)
            counts = counts + 1;
        else
            misclass = misclass + 1;
        end
    end
    corr_preds_Z1F2(i) = counts;
    incorr_preds_Z1F2(i) = misclass;
end

classify_accZ1F2 = zeros(1,5);
Error_ratesZ1F2 = zeros(1,5);
for i = 1:5
    classify_accZ1F2(i) = (corr_preds_Z1F2(i)/900)*100;
    Error_ratesZ1F2(i) = (incorr_preds_Z1F2(i)/900)*100;
end

diff_matZ1F2 = Z1F2_normal - original_index;
error_valsZ1F2 = diff_matZ1F2==0;
classification_accuracyZ1F2 = (sum(error_valsZ1F2(:))/4500)*100;
error_rateZ1F2 = 100-classification_accuracyZ1F2