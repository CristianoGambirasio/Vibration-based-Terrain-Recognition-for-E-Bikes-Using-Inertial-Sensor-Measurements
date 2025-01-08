function [outData] = create_features(inData)
%CREATE_FEATURES
%   Transform a table of measurements in features for the model
% 
% INPUTS
% inData=
%     table containing measurements from a correct number of wheel
%     revolution (3 revolution)
%
% OUTPUT
% outData =
%     table with height 1 ready for predict() method

preprocessed_data = inData;

% Same preprocessing done during model training
LSB_g_ACC = 4096;
LSB_deg_s_GYRO = 16.4;

% Accelerations conversion (g)
preprocessed_data.nicla_accX = preprocessed_data.nicla_accX./LSB_g_ACC;
preprocessed_data.nicla_accY = preprocessed_data.nicla_accY./LSB_g_ACC;
preprocessed_data.nicla_accZ = preprocessed_data.nicla_accZ./LSB_g_ACC;

% Gyroscope conversion (°/s)
preprocessed_data.nicla_gyroY = preprocessed_data.nicla_gyroY./LSB_deg_s_GYRO;

% Motor speed measurements are in 10*km/h
preprocessed_data(:,{'motor_spd'})=preprocessed_data(:,{'motor_spd'})./10;

% Rotation matrix used during training
R = [0.936492845966004	0	-0.350686682744717;
                    0	1	0;
    0.350686682744717	0	0.936492845966004];


preprocessed_data(:,1:3) = array2table((R * preprocessed_data{:,1:3}')');
preprocessed_data(:,4:6) = array2table((R * preprocessed_data{:,4:6}')');

% Removing gravity from accelerometer's z-axes 
preprocessed_data.nicla_accZ = preprocessed_data.nicla_accZ +1;

% Normalizing inertial measurements by speed
preprocessed_data.nicla_accX = preprocessed_data.nicla_accX ./ preprocessed_data.motor_spd;
preprocessed_data.nicla_accY =preprocessed_data.nicla_accY ./preprocessed_data.motor_spd;
preprocessed_data.nicla_accZ =preprocessed_data.nicla_accZ ./preprocessed_data.motor_spd;

preprocessed_data.nicla_gyroY =preprocessed_data.nicla_gyroY ./preprocessed_data.motor_spd;

% Norm of accelerations
preprocessed_data.norm_acc = sqrt(preprocessed_data.nicla_accX.^2 + ...
    preprocessed_data.nicla_accY.^2+ preprocessed_data.nicla_accZ.^2);

%% Computing time features

% Same threshold used for training
th1=[-0.02 -0.015 -0.15 0.006];
th2=[0.007 0.005 0.15 0.01];
th3 =[-0.04 -0.025 -0.25 0.012];
th4 = [0.015  0.01  0.25 0.02];
% inertials measurements and statistics used by the model
inertials = {'nicla_accX','nicla_accZ', 'nicla_gyroY', 'norm_acc'};
statistics = {'mean_','var_','max_','min_','th1_cross_','th2_cross_','th3_cross_','th4_cross_'};

grouped_stats = mean(preprocessed_data{:,inertials}); % Mean computation

grouped_stats = [grouped_stats var(preprocessed_data{:,inertials})]; % Variance computation

grouped_stats = [grouped_stats max(preprocessed_data{:,inertials})]; % Maximum computation

grouped_stats = [grouped_stats min(preprocessed_data{:,inertials})]; % Minimum computation

grouped_stats = [grouped_stats thresh_crossing(preprocessed_data{:,inertials},th1)]; % Threshold crossing computation

grouped_stats = [grouped_stats thresh_crossing(preprocessed_data{:,inertials},th2)];

grouped_stats = [grouped_stats thresh_crossing(preprocessed_data{:,inertials},th3)];

grouped_stats = [grouped_stats thresh_crossing(preprocessed_data{:,inertials},th4)];

% Creating same var names as training model
var_names = cell(length(inertials) * length(statistics), 1);
k=1;
for i = 1:length(statistics)
    for j= 1:length(inertials)
        var_names{k} = strcat(statistics{i}, inertials{j});
        k=k+1;
    end  
end

% Renaming column
grouped_stats = array2table(grouped_stats,'VariableNames',var_names);

% Threshold crossing function (+ handling of low number measurements case)
function crossing = thresh_crossing(x,means)
    if(height(x)<2)
        crossing = [0,0,0,0,0,0,0,0];
        return
    end
    diffs = x - means;
    signs = sign(diffs);
    sign_changes = diff(signs);

    th_cross = abs(sign_changes) == 2;

    crossing = sum(th_cross);
end

%% Computing frequency features
% GyroY
X = preprocessed_data.nicla_gyroY;
N = 200;
Y = fft(X,N);
Y = abs(Y(1:N/2+1));
FFTGyroY = Y';

%AccX
X = preprocessed_data.nicla_accX;
N = 200;
Y = fft(X,N);
Y = abs(Y(1:N/2+1));
FFTAccX = Y';

%AccZ
X = preprocessed_data.nicla_accZ;
N = 200;
Y = fft(X,N);
Y = abs(Y(1:N/2+1));
FFTAccZ = Y';

% Handling low number of measurement case
if(height(inData)<2)
    FFTGyroY = zeros(1,51);
    FFTAccX = zeros(1,51);
    FFTAccZ = zeros(1,51);
end

% Computing frequency features
gyroY = [sum(FFTGyroY(:,2:7),2),sum(FFTGyroY(:,7:15),2)];
accX = [sum(FFTAccX(:,9:15),2) sum(FFTAccX(:,37:47),2) sum(FFTAccX(:,49:end),2) ];%mean(FFTAccX,2)];
accZ = [sum(FFTAccZ(:,5:11),2) sum(FFTAccZ(:,61:end),2)];

% Renaming features as during training model
freq_features = array2table([gyroY accX accZ],"VariableNames",... % Frequency features
    {'GyroY1-3Hz','GyroY3-7Hz','AccX4-7Hz','AccX18-23Hz','AccX24-50Hz','AccZ2-5Hz','AccZ30-50Hz'});

%% Combining features 
outData = [grouped_stats freq_features];
end

