% Script designed to simulate the model's behavior in an online setting, 
% utilizing an offline collected dataset as input

%% Importing data
% Damper dataset
addpath('functions\');
data = readtable("sensor_data\Test\TEST_Damper_data_2024_09_25_15_29_35.csv");
label = load("sensor_data\Test\label_Damper.mat").label;

% No damper dataset
% data = readtable("sensor_data\Test\TEST_NoDamper_data_2024_09_25_15_46_15.csv");
% label = load("sensor_data\Test\label_NoDamper.mat").label;

% Creating groups every 3 wheel rotation
data = create_rev_groups(data,0.38,30); % Already remove NAN and 0 speed measurements
                                        % 30 revs because motor speed is not yet converted
% data.timestamp = datetime(data.timestamp,'ConvertFrom','posixtime','TimeZone','Europe/Rome');
measurements = {'nicla_accX','nicla_accY','nicla_accZ', 'nicla_gyroX', 'nicla_gyroY', 'nicla_gyroZ','motor_spd'};
grouped_values = splitapply(@(x) {array2table(x,"VariableNames",measurements)}, data{:,measurements}, data.group);

clear("measurements");

%% Online prediction simulation
% Model parameters
HMM_SEQUENCE_LENGTH = 20; % Number of previous prediction considered in HMM
MEAN_POSTERIOR_WINDOW = 5; % Number of previous posterior probabilities considered to compute mean posterior probability
MIN_PROBABILITY_THRESHOLD = 0.75; % Minimum mean posterior probability to accept model's prediction

% Initializing transition and emission matrix for HMM
trans_matrix = [.92 .04 .04;.04 .92 .04;.04 .04 .92];
emis_matrix = [.5 .25 .25; .25 .5 .25 ;.25 .25 .5 ];

% Loading the SVM model
%svm_model = load("models\SVMModel.mat").svm_model;
svm_model = load("models\SVMModel_select.mat").svm_model;

svm_pred_list = []; % list of svm prediction
hmm_pred_list = []; % list of prediction after hmm

posterior_list = []; % list of posterior probabilities of svm model
mean_posterior_list = []; % list of mean posterior probabilities
tic;
% profile on
for i=1:height(grouped_values)
    
    
    sample = create_features(grouped_values{i}); %Create features from i-th group of the test

    % Predicting using SVM model
    [svm_pred, ~,~,posterior] = predict(svm_model,sample);
    posterior_list = [posterior_list; posterior]; %Add posterior to posteriors list

    % Computing mean posterior
    if(i>MEAN_POSTERIOR_WINDOW)
        mean_posterior = mean(posterior_list(i-MEAN_POSTERIOR_WINDOW:i,svm_pred));
    else
        mean_posterior = mean(posterior_list(1:i,svm_pred));
    end

    mean_posterior_list = [mean_posterior_list; mean_posterior]; % Add mean posterior to mean posteriors list
   
    % Dynamic emission matrix based on mean posterior probability
    emis_matrix = [mean_posterior       (1-mean_posterior)/2    (1-mean_posterior)/2;
                  (1-mean_posterior)/2  mean_posterior          (1-mean_posterior)/2;
                  (1-mean_posterior)/2  (1-mean_posterior)/2    mean_posterior];

    
    svm_pred_list = [svm_pred_list svm_pred]; % Add svm prediction to svm predictions list
    % Enhancing predictions using HMM
    if(i>HMM_SEQUENCE_LENGTH)
        hmm_pred_seq = (hmmviterbi(svm_pred_list(i-HMM_SEQUENCE_LENGTH:i),trans_matrix,emis_matrix));
    else
        hmm_pred_seq = hmmviterbi(svm_pred_list(1:i), trans_matrix,emis_matrix);
    end

    % Preventing changes for predictions with too low probability
    if i==1
        hmm_pred_list = [hmm_pred_list hmm_pred_seq(end)];
    elseif (mean_posterior < MIN_PROBABILITY_THRESHOLD)
        hmm_pred_list = [hmm_pred_list hmm_pred_list(i-1)];
    else
        hmm_pred_list = [hmm_pred_list hmm_pred_seq(end)];
    end
end
% profile off
% profile viewer
elapsed_time = toc;


predictions = hmm_pred_list;
predictions = svm_pred_list;

disp([num2str(elapsed_time), ' s'])
disp([num2str(elapsed_time/height(grouped_values)), ' s/prediction'])

clear("elapsed_time"); clear("HMM_SEQUENCE_LENGTH"); clear("i"); clear("mean_posterior"); clear("MIN_PROBABILITY_THRESHOLD");
clear("posterior"); clear("MEAN_POSTERIOR_WINDOW"); clear("emis_matrix");
clear("hmm_pred_seq"); clear("sample"); clear("svm_model"); clear("svm_pred"); clear("trans_matrix"); clear("svm_pred_list");
clear("hmm_pred_list")

%% Plotting posteriors
% posteriors of SVM classifier and mean posteriors 
figure('NumberTitle', 'off', 'Name', 'Raw SVM model posteriors');
plot(posterior_list);
legend("Asphalt", "Gravel", "Cobblestone","Location","southeast")
ylabel("Probability")
title ("Raw posterior probabilities")

figure('NumberTitle', 'off', 'Name', 'MovMean SVM model posteriors');
plot(movmean(posterior_list,[5,0]));
legend("Asphalt", "Gravel", "Cobblestone","Location","southeast")
ylabel("Probability")
title ("Smoothed posterior probabilities")

%% Plotting results  
% Confusion Matrix
figure('NumberTitle', 'off', 'Name', 'Online simulation - Confusion Matrix');
classNames = {'Asphalt', 'Gravel', 'Cobblestone'};
confusionchart(classNames(label), classNames(predictions))
title("Confusion Matrix on Test Dataset 2")
disp(strcat("SVM Accuracy: ", string(nnz(predictions'==label)/length(predictions))))

% Predictions sequence
[idx,terrain,terrain_names, color] = get_label(label);
figure('NumberTitle', 'off', 'Name', 'Online simulation - Predictions');
yyaxis left
hold on
for i =1:length(terrain)
    patch([idx(i) idx(i+1)-1 idx(i+1)-1 idx(i)],[1 1 3 3],color(terrain(i),:),'DisplayName',terrain_names(terrain(i)),...
        'FaceAlpha',0.1, 'EdgeColor','none');
end
a = scatter(1:length(predictions),predictions,150,'|','LineWidth',2,'MarkerFaceAlpha',0.1, 'DisplayName',"Predicted Terrain");
ylabel('Class')
yticks([1,2,3]);
yticklabels({'Asphalt',"Gravel", "Cobblestone"});
yyaxis right
plot(movmean(posterior_list(:,1),[5,0]),"-",'DisplayName',"Asphalt SVM Prediction probability","Color",[0    0.4470    0.7410])
hold on
plot(movmean(posterior_list(:,2),[5,0]),"-",'DisplayName',"Gravel SVM Prediction probability","Color",[0.8500    0.3250    0.0980])
hold on
plot(movmean(posterior_list(:,3),[5,0]),"-",'DisplayName',"Cobblestone SVM Prediction probability","Color",[0.9290    0.6940    0.125])
set(gca, 'SortMethod', 'depth')
ylabel('Posterior Probability')
xlabel('Sample')
legend('Location','eastoutside')

clear("idx");clear("i"); clear("terrain_names");clear("terrain");clear("color");clear("a");


function [idx, terr_num, terr_name, color] = get_label(label)
    terr_name = ["Label Asphalt","Label Gravel","Label Cobblestone"];
    terr_num = [label(1)];
    idx = 1;
    color = [0 0 1; 1 0 0; 1 0.5 0];
    for i = 2:length(label)
        if label(i) ~= terr_num(end)
            idx = [idx, i];
            terr_num = [terr_num, label(i)];
        end
    end
    idx = [idx, length(label)];   
end