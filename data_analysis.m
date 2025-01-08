%% Loading data
addpath('functions\');
terrains = [{'Asphalt'},{'Gravel'},{'Cobblestone'}];
file_names = {};
for i = 1:length(terrains) % Names of file in sensor data terrain folders
    file_names{i} = strcat(terrains{i},"/",{dir(fullfile(strcat("sensor_data/",char(strcat(terrains{i},"/")), '*.csv'))).name});
end
raw_data = [];
for i=1:length(terrains)
    temp_data = table();
    
    for j=1:length(file_names{i})
        temp_table = readtable(strcat("sensor_data/",file_names{i}{j}),'VariableNamingRule','modify');
    
        temp_data = [temp_data; temp_table];
    end
    raw_data = [raw_data {temp_data}]; % 1*n cell of different terrain data tables 
end

clear("j");clear("temp_table");clear("file_names");clear("i");clear("temp_data");
%% Data preprocessing
SHOW_PLOT = 0; % plots for redundant measurements choice    0 = No plots; 1 = plots
PRINT_NAN = 0; % NaN values at the end of preprocessing     0 = No NaN print; 1 = NaN print

preprocessed_data = raw_data;

LSB_g_ACC = 4096;
LSB_deg_s_GYRO = 16.4;

for i= 1:length(raw_data)

    % Accelerations conversion (g)
    preprocessed_data{i}.nicla_accX = preprocessed_data{i}.nicla_accX./LSB_g_ACC;
    preprocessed_data{i}.nicla_accY = preprocessed_data{i}.nicla_accY./LSB_g_ACC;
    preprocessed_data{i}.nicla_accZ = preprocessed_data{i}.nicla_accZ./LSB_g_ACC;
    
    % Gyroscope conversion (°/s)
    preprocessed_data{i}.nicla_gyroX = preprocessed_data{i}.nicla_gyroX./LSB_deg_s_GYRO;
    preprocessed_data{i}.nicla_gyroY = preprocessed_data{i}.nicla_gyroY./LSB_deg_s_GYRO;
    preprocessed_data{i}.nicla_gyroZ = preprocessed_data{i}.nicla_gyroZ./LSB_deg_s_GYRO;
    
    % Dropping useless column
        %nicla_ts       -> relative internal clock from accelerometer
        %nicla_roll     -> not reliable from BRICK documentation
        %nicla_pitch    -> not reliable from BRICK documentation
        %gps_time       -> info already available as "timestamp"
    preprocessed_data{i}(:,{'nicla_ts','nicla_roll','nicla_pitch','gps_time'})=[];
    
    % UNIX to Europe/Rome timestamp conversion
    preprocessed_data{i}.timestamp = datetime(preprocessed_data{i}.timestamp,'ConvertFrom','posixtime','TimeZone','Europe/Rome');
    
    % Speed measurament choice (motor speed)
    if(SHOW_PLOT && i==1)
        figure;
        plot(preprocessed_data{i}.gps_spd)
        hold on
        plot(preprocessed_data{i}.motor_spd./10)
        title('GPS and Motor speed measurement')
        xlabel('Time')
        ylabel('Speed [km/h]')
        legend('GPS measurement','Motor measurement')
    end
    preprocessed_data{i}(:,{'gps_spd'})=[];

    % Speed measurement are collected as 10*km/h
    preprocessed_data{i}(:,{'motor_spd'})=preprocessed_data{i}(:,{'motor_spd'})./10;
    
    % Altitude measurament choice (GPS altitude)
    if(SHOW_PLOT && i==1)
        figure;
        plot(preprocessed_data{i}.nicla_alt)
        hold on
        plot(preprocessed_data{i}.gps_alt)
        title('Nicla and GPS altitude measurement')
        xlabel('Time')
        ylabel('Altitude [m]')
        ylim([0,300])
        legend('Nicla measurement','GPS measurement')
    end
    preprocessed_data{i}(:,{'nicla_alt'})=[];
    
    % Motor error checking -> error print for column "motor_err"
    for j = 0:7
        err_col_name = strcat('motor_err_',string(j));
        if(any(preprocessed_data{i}.(err_col_name) ~= 0 & ~isnan(preprocessed_data{i}.(err_col_name))))
            error_message = strcat('Motor error| error name:',string(j));
            error(error_message)
            continue;
        end
        preprocessed_data{i}.(err_col_name)=[];
    end

    %Inertial measurement rotation
    static = [0.352, 0, 0.940]; %Accelerometer measurement in static situation -> check before during experiments
    gravity = [0, 0, 1];

    theta = cross(static,gravity); % Rotational vector
    alpha = acos(dot(static,gravity)/(norm(static)*norm(gravity))); %tilted 20.5°

    normalized_theta = theta / norm(theta);

    K = [0,                     -normalized_theta(3),   normalized_theta(2); % skew-symmetric theta
        normalized_theta(3),    0,                      -normalized_theta(1);
        -normalized_theta(2),   normalized_theta(1),    0];

    R = eye(3) + sin(alpha) * K + (1 - cos(alpha)) * (K*K); % Rotation Matrix

    % Rotatin inertial measurements
    preprocessed_data{i}(:,2:4) = array2table((R * table2array(preprocessed_data{i}(:,2:4))')');
    preprocessed_data{i}(:,5:7) = array2table((R * table2array(preprocessed_data{i}(:,5:7))')');

    % Removing gravity from accelerometer's z-axes 
    preprocessed_data{i}.nicla_accZ = preprocessed_data{i}.nicla_accZ +1;
        
    % Removing NaN speed measurement and NaT timestamp 
    preprocessed_data{i} = preprocessed_data{i}(~isnan(preprocessed_data{i}.motor_spd),:);
    %preprocessed_data{i} = preprocessed_data{i}(~isnat(preprocessed_data{i}.timestamp),:);

    % Normalizing inertial measurements by speed
    preprocessed_data{i}.nicla_accX = preprocessed_data{i}.nicla_accX ./ preprocessed_data{i}.motor_spd;
    preprocessed_data{i}.nicla_accY =preprocessed_data{i}.nicla_accY ./preprocessed_data{i}.motor_spd;
    preprocessed_data{i}.nicla_accZ =preprocessed_data{i}.nicla_accZ ./preprocessed_data{i}.motor_spd;

    preprocessed_data{i}.nicla_gyroX = preprocessed_data{i}.nicla_gyroX ./ preprocessed_data{i}.motor_spd;
    preprocessed_data{i}.nicla_gyroY =preprocessed_data{i}.nicla_gyroY ./preprocessed_data{i}.motor_spd;
    preprocessed_data{i}.nicla_gyroZ =preprocessed_data{i}.nicla_gyroZ ./preprocessed_data{i}.motor_spd;

    %Print number of NaN
    if (PRINT_NAN)
        for j = 2:width(preprocessed_data{i})
            if(j~=13) %column of char
                disp(strcat(terrains{i}," ",preprocessed_data{i}.Properties.VariableNames{j},":", ...
                    string(sum(isnan(table2array(preprocessed_data{i}(:,j)))))))
            end
        end
    end

    % Norm of accelerations
    preprocessed_data{i}.norm_acc = sqrt(preprocessed_data{i}.nicla_accX.^2 + ...
        preprocessed_data{i}.nicla_accY.^2+ preprocessed_data{i}.nicla_accZ.^2);

    % Train test split
    experiments_idx = find(diff(preprocessed_data{i}.timestamp) > seconds(30) | diff(preprocessed_data{i}.timestamp) < seconds(-30));
    experiments_idx = [0; experiments_idx; height(preprocessed_data{i})];
    preprocessed_data{i}.test = zeros([height(preprocessed_data{i}),1]);

    for j=1:length(experiments_idx)-1
        experiment = preprocessed_data{i}(experiments_idx(j)+1:experiments_idx(j+1),:);
        train_idx = floor(0.7*height(experiment));
        preprocessed_data{i}.test(experiments_idx(j)+1+train_idx:experiments_idx(j+1)) = 1;
    end

end

clear('error_message');clear('i');clear('LSB_deg_s_GYRO');clear("LSB_g_ACC");clear('SHOW_PLOT');clear('err_col_name');clear("j");
clear("PRINT_NAN");clear("experiment"); clear("experiments_idx");
clear("alpha"); clear("gravity"); clear("K"); clear("normalized_theta"); clear("R"); clear("static"); clear("theta");
clear("train_idx")

%% Data visualization
graphs(terrains,preprocessed_data,0,0,0,1) % Custom data visualization function

%% Groups assignment based on number of wheel revolutions
preprocessed_data = create_rev_groups(preprocessed_data,0.38,3);

%% Grouping and computing statistics
% Grouping data and computing statistics (time features)
measurements = {'nicla_accX','nicla_accZ', 'nicla_gyroY','norm_acc'}; % useful measurements for modeling

grouped_data = {};
for i=1:length(preprocessed_data)
    
    % Mean of measurements
    grouped_data{i} = array2table(splitapply(@(x)mean(x),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('mean_',measurements));
    
    % Variance of measurements
    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x)var(x),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('var_',measurements))];

    % Max value of measurements
    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x)max(x),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('max_',measurements))];
    
    % Min value of measurements
    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x)min(x),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('min_',measurements))];
    
    % Threshold crossing counting
    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x) ...
        thresh_crossing(x,[-0.02 -0.015 -0.15 0.006]),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('th1_cross_',measurements))];

    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x) ...
        thresh_crossing(x,[0.007 0.005 0.15 0.01]),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('th2_cross_',measurements))];

    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x) ...
        thresh_crossing(x,[-0.04 -0.025 -0.25 0.012]),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('th3_cross_',measurements))];

    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x) ...
        thresh_crossing(x,[0.015  0.01  0.25 0.02]),preprocessed_data{i}{:,measurements},preprocessed_data{i}.group), ...
        "VariableNames",strcat('th4_cross_',measurements))];

    % Mode of "test" feature -> to choose if group is test or train
    grouped_data{i} = [grouped_data{i} array2table(splitapply(@(x)mode(x),preprocessed_data{i}{:,'test'},preprocessed_data{i}.group), ...
        "VariableNames","test")];

    % Label
    grouped_data{i}.label(:) = i;

    % Values as array of groups of measurements
    grouped_values{i} = splitapply(@(x) {array2table(x,"VariableNames",measurements)}, preprocessed_data{i}{:,measurements}, preprocessed_data{i}.group); 
end

% Threshold crossing function
function crossing = thresh_crossing(x,means)
    diffs = x - means;
    signs = sign(diffs);
    sign_changes = diff(signs);

    th_cross = abs(sign_changes) == 2;

    crossing = sum(th_cross);
end


clear("i");clear("measurements");

%% FFTs computation 
% Computing FFTs of every measurements group

for i=1:length(preprocessed_data)
    FFTsGyroY{i} = compute_FFT(grouped_values{i},'nicla_gyroY'); % FFTs of every sample
    FFTsAccX{i} = compute_FFT(grouped_values{i},'nicla_accX');
    FFTsAccZ{i} = compute_FFT(grouped_values{i},'nicla_accZ');
end

FFTs = {FFTsGyroY, FFTsAccX, FFTsAccZ}; % GyroY, AccX, AccZ FFTs

% FFT computing function
function result = compute_FFT(grouped_values,feature)
    result = [];
    for j = 1:height(grouped_values)
        X = grouped_values{j}.(feature);

        N = 200;
        Y = fft(X,N);
        Y = abs(Y(1:N/2+1));

        result = [result; Y'];
    end
end

clear("FFTsAccZ"); clear("FFTsAccX"); clear("FFTsGyroY"); clear("i");

%% MODELING -----------------------------------------------------------------------------------------------

%% train/test datasets creation (ONLY TIME FEATURE)

model_features = grouped_data{1}.Properties.VariableNames; % Selected features

% Finding train/test indexes
test_idx=[];
train_idx=[];
features = [];
for i=1:length(grouped_data)
    test_idx = logical([test_idx; grouped_data{i}.test==1]);
    train_idx = logical([train_idx; grouped_data{i}.test==0]);
    features = [features; grouped_data{i}];
end

train_set = features(train_idx,model_features);
test = features(test_idx,model_features);

train_set.test=[];
test.test = [];

clear("test_idx");clear("features"); clear("model_features"); clear("train_idx"); clear("i")

%% train/test datasets creation (ONLY FREQUENCY FEATURE)

% Finding train/test indexes
test_idx=[];
train_idx=[];
label = [];
for i=1:length(grouped_data)
    test_idx = logical([test_idx; grouped_data{i}.test==1]);
    train_idx = logical([train_idx; grouped_data{i}.test==0]);
    label = [label; grouped_data{i}(:,end)];
end

% Extracting frequency features
gyroY = [];
accX = [];
accZ = [];
for i=1:length(grouped_data)
    gyroY = [gyroY; sum(FFTs{1}{i}(:,2:7),2),sum(FFTs{1}{i}(:,7:15),2)];
    accX = [accX; sum(FFTs{2}{i}(:,9:15),2) sum(FFTs{2}{i}(:,37:47),2) sum(FFTs{2}{i}(:,49:end),2)];
    accZ = [accZ; sum(FFTs{3}{i}(:,5:11),2) sum(FFTs{3}{i}(:,61:end),2)];

end

train_set = array2table([gyroY(train_idx,:) accX(train_idx,:) accZ(train_idx,:)],"VariableNames",...
    {'GyroY1-3Hz','GyroY3-7Hz','AccX4-7Hz','AccX18-23Hz','AccX24-50Hz','AccZ2-5Hz','AccZ30-50Hz'});
test = array2table([gyroY(test_idx,:) accX(test_idx,:) accZ(test_idx,:)],"VariableNames",...
    {'GyroY1-3Hz','GyroY3-7Hz','AccX4-7Hz','AccX18-23Hz','AccX24-50Hz','AccZ2-5Hz','AccZ30-50Hz'});

% Adding labels
train_set.label = table2array(label(train_idx,:));
test.label = table2array(label(test_idx,:));


clear("test_idx"); clear("train_idx"); 
clear("gyroY"); clear("accX"); clear("accZ"); clear("label"); clear("i");

%% train/test datasets creation (TIME + FREQUENCY FEATURE)

% Finding train/test indexes
test_idx=[];
train_idx=[];
features = [];
for i=1:length(grouped_data)
    test_idx = logical([test_idx; grouped_data{i}.test==1]);
    train_idx = logical([train_idx; grouped_data{i}.test==0]);
    features = [features; grouped_data{i}];
end

train_t = features(train_idx,:); % Time features
test_t = features(test_idx,:);

% Extracting frequency features
gyroY = [];
accX = [];
accZ = [];
for i=1:length(grouped_data)
    gyroY = [gyroY; sum(FFTs{1}{i}(:,2:7),2),sum(FFTs{1}{i}(:,7:15),2)];
    accX = [accX; sum(FFTs{2}{i}(:,9:15),2) sum(FFTs{2}{i}(:,37:47),2) sum(FFTs{2}{i}(:,49:end),2)];
    accZ = [accZ; sum(FFTs{3}{i}(:,5:11),2) sum(FFTs{3}{i}(:,61:end),2)];
end

train_f = array2table([gyroY(train_idx,:) accX(train_idx,:) accZ(train_idx,:)],"VariableNames",... % Frequency features
    {'GyroY1-3Hz','GyroY3-7Hz','AccX4-7Hz','AccX18-23Hz','AccX24-50Hz','AccZ2-5Hz','AccZ30-50Hz'});
test_f = array2table([gyroY(test_idx,:) accX(test_idx,:) accZ(test_idx,:)],"VariableNames",...
    {'GyroY1-3Hz','GyroY3-7Hz','AccX4-7Hz','AccX18-23Hz','AccX24-50Hz','AccZ2-5Hz','AccZ30-50Hz'});

% Joining time and frequency features
train_set = [train_t train_f];
test = [test_t test_f];
train_set.test=[];
test.test = [];

clear("train_t"); clear("test_t"); clear("train_f"); clear("test_f"); 
clear("test_idx");clear("features"); clear("train_idx"); clear("i");
clear("gyroY"); clear("accX"); clear("accZ")

%% Feature selection
% Selection using lasso penalization
s=templateLinear("Regularization","lasso","Learner","svm","lambda",0.05); 
% Standardizing dataset
train_std = removevars(train_set, {'label'});
train_std = normalize(train_std);
train_std.label = train_set.label;

test_std = removevars(test, {'label'});
test_std = normalize(test_std);
test_std.label = test.label;

lasso_model = fitcecoc(train_std, "label" ,'Learners',s,'Coding','ternarycomplete'); 

% Lasso model evaluation (lambda tuning) lambda = 0.05
% accuracies =[];
% for i = 1:length(lambdas)
%     s=templateLinear("Regularization","lasso","Learner","svm","lambda",lambdas(i)); 
%     lasso_model_t = fitcecoc(train_std, "label" ,'Learners',s,'Coding','ternarycomplete'); 
%     lasso_pred = predict(lasso_model_t,test_std); % Performance on test set
%     accuracies = [accuracies nnz(lasso_pred==test_std.label)/length(lasso_pred)];
% end
% 
% plot(lambdas,accuracies)
% title("λ Tuning")
% ylabel("Accuracy")
% xlabel("λ")
% xline([0.05],"Color","r")
% legend(["Model Accuracy", "Chosen λ"])

% figure;
% confusionchart(test_std.label, lasso_pred)
% title("Confusion Matrix")
% disp(strcat("SVM Accuracy: ", string(nnz(lasso_pred==test_std.label)/length(lasso_pred))))


% Computing feature importance after lasso regularization
binary_learners  = lasso_model.BinaryLearners;

betas = [];
for i = 1:length(binary_learners)
    betas = [betas abs(binary_learners{i}.Beta)];
end

betas = rescale(betas,"InputMin",min(betas),"InputMax",max(betas))'; % Rescaling

[~, ord_idx] = sort(mean(betas), 'descend'); % Ordering

figure; % Learners feature importance
bar(strrep(lasso_model.PredictorNames(ord_idx),"_"," "),betas(:,ord_idx))
ylim([0,1])
legend('Gravel vs Asphalt','Cobblestone vs Asphalt/Gravel','Cobblestone vs Gravel','Gravel vs Cobblestone/Asphalt', ...
    'Cobblestone vs Asphalt','Asphalt vs Cobblestone/Gravel')
xtickangle(90)
title("Feature Importance")
subtitle("Binary learner importance")

figure; % Mean feature importance
y = mean(betas);
bar(strrep(lasso_model.PredictorNames(ord_idx),"_"," "),y(ord_idx))
ylim([0,1])
xtickangle(90)
title("Feature Importance")
subtitle("Mean Importance")

rem_features = lasso_model.PredictorNames(y<0.005); % useles features

% Removing useless features
model_features = train_set.Properties.VariableNames; % Selected features
model_features = setdiff(model_features, rem_features,'stable'); % Removed features

train_select = train_set(:,model_features);% Dataset with data selection
test_select = test(:,model_features);

clear("betas"); clear("binary_learners"); clear("i");clear("lasso_model"); clear("y");
clear("ord_idx"); clear("s"); clear("train_std"); clear("test_std"); clear("rem_features"); clear("model_features");
clear("train_set"); clear("test");

%% SVM Crossvalidation
s=templateSVM('Standardize',true ,'KernelFunction','linear','OutlierFraction',0.05);
params = hyperparameters("fitcecoc",train_select,train_select.label,"svm");
params(2).Range = [1e-3,100];
params = params(2);
svm_tuning_model = fitcecoc(train_select, "label" ,'Learners',s,'Coding','ternarycomplete',...
    'OptimizeHyperparameters',params,'HyperparameterOptimizationOptions',struct("Kfold",3),'Prior','uniform'); % Multiclass SVM

%% SVM Model
s=templateSVM('Standardize',true ,'KernelFunction','linear','OutlierFraction',0.05); % SVM with standardization of features and linear kernel
svm_model = fitcecoc(train_select, "label" ,'Learners',s,'Coding','ternarycomplete','Prior','uniform'); % Multiclass SVM
svm_pred = predict(svm_model,test_select); % Performance on test set

figure;
confusionchart(test_select.label, svm_pred)
title("Confusion Matrix") 
disp(strcat("SVM Accuracy: ", string(nnz(svm_pred==test_select.label)/length(svm_pred))))

% Retrain model on whole dataset + Fit posterior probabilities
%svm_model = fitcecoc([train_select;test_select], "label" ,'Learners',s,'Coding','ternarycomplete','FitPosterior',true,"Verbose",2); 

clear("s")
%% Test for SVM + HMM approach

% Transition and emission matrix
trans_init = [.94 .03 .03;.03 .94 .03;.03 .03 .94];
emis_init = [.80 .1 .1; .1 .80 .1 ;.1 .1 .80 ];

svm_hmm_pred = hmmviterbi(svm_pred,trans_init,emis_init);

confusionchart(test_select.label, svm_hmm_pred)
disp(strcat("SVM + HMM Accuracy: ", string(nnz(svm_hmm_pred'==test_select.label)/length(svm_hmm_pred))))

clear("emis_init"); clear("emis_estimated"); clear("train_predict"); clear("trans_estimated"); clear("trans_init")

%% Feature importance
% 6 binary learners
%   1 -> Gravel vs Asphalt
%   2 -> Cobblest vs Asphalt_or_Gravel
%   3 -> Cobblestone vs Gravel
%   4 -> Gravel vs Cobblestone_or_Asphalt
%   5 -> Cobblestone vs Asphalt
%   6 -> Asphalt vs Cobblestone_or_Gravel

% Feature importance = mean of beta coefficient in binary learners
% rescaled to have values between [0 1]
% Weighted with learner weights

binary_learners  = svm_model.BinaryLearners;

betas = [];
for i = 1:length(binary_learners)
    betas = [betas abs(binary_learners{i}.Beta)];
end

betas = rescale(betas,"InputMin",min(betas),"InputMax",max(betas))'; % Rescaling

[~, ord_idx] = sort(mean(betas), 'descend'); % Ordering

figure; % Learners feature importance
bar(strrep(svm_model.PredictorNames(ord_idx),"_"," "),betas(:,ord_idx))
ylim([0,1])
legend('Gravel vs Asphalt','Cobblestone vs Asphalt/Gravel','Cobblestone vs Gravel','Gravel vs Cobblestone/Asphalt', ...
    'Cobblestone vs Asphalt','Asphalt vs Cobblestone/Gravel')
xtickangle(90)
title("Feature Importance")
subtitle("Binary learner importance")

figure; % Mean feature importance
y = mean(betas);
bar(strrep(svm_model.PredictorNames(ord_idx),"_"," "),y(ord_idx))
ylim([0,1])
xtickangle(90)
title("Feature Importance")
subtitle("Mean Importance")

clear("betas"); clear("binary_learners"); clear("i");clear(""); clear("y");
