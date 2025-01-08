%% Loading data
addpath('functions\');
terrains = [{'Asphalt'},{'Gravel'},{'Cobblestone'}];
file_names = {};
for i = 1:length(terrains) % Names of file in sensor data terrain folders
    file_names{i} = strcat(terrains{i},"/",{dir(fullfile(strcat("sensor_data/",char(strcat(terrains{i},"/")), '*.csv'))).name});
end
divided_experiments_data = [];
for i=1:length(terrains)
    %temp_data = table();
    temp_data = [];
    for j=1:length(file_names{i})
        temp_table = readtable(strcat("sensor_data/",file_names{i}{j}),'VariableNamingRule','modify');
        temp_data = [temp_data {temp_table}];
    end
    divided_experiments_data = [divided_experiments_data {temp_data}]; % raw data divided by experiment
end

%Standardizing number of experiment of cobblestone (missing experiments) with gravel and asphalt
divided_experiments_data{3}{8} = divided_experiments_data{3}{5};
divided_experiments_data{3}{7} = divided_experiments_data{3}{4};
divided_experiments_data{3}{6} = divided_experiments_data{3}{3};
divided_experiments_data{3}{3} = divided_experiments_data{3}{2};
divided_experiments_data{3}{5} = [];
divided_experiments_data{3}{4} = [];
divided_experiments_data{3}{2} = [];

clear("j");clear("temp_table");clear("file_names");clear("i");clear("temp_data");

%% Experiment preprocessing
% experiment picking function load the specified experiment in every
% terrain and do the preprocessing

single_experiment_data = experiment_picking(divided_experiments_data,1); %test function on first esperiment

function single_experiment_data = experiment_picking(divided_experiments_data,n_experiment)
    single_experiment_data = [];
    
    for i= 1:length(divided_experiments_data)
        if (istable(divided_experiments_data{i}{n_experiment})) % Check if experiment is in the folder
            single_experiment_data = [single_experiment_data {divided_experiments_data{i}{n_experiment}}];
        else % Empty table
            single_experiment_data = [single_experiment_data {table('Size', [0, 40],...
                'VariableTypes', repmat({'double'}, 1, 40),...
                'VariableNames',divided_experiments_data{1}{1}.Properties.VariableNames)}];
            continue;
        end
    end

    % Preprocessing parameters
    LSB_g_ACC = 4096; 
    LSB_deg_s_GYRO = 16.4;

    % Rotation matrix computed in file data_analysis_and_model_train
    R = [0.936492845966004	0	-0.350686682744717;
                    0	1	0;
         0.350686682744717	0	0.936492845966004];

   
    for i= 1:length(single_experiment_data) % Preprocess the experiment on each terrain
        % Accelerations conversion (g)
        single_experiment_data{i}.nicla_accX = single_experiment_data{i}.nicla_accX./LSB_g_ACC;
        single_experiment_data{i}.nicla_accY = single_experiment_data{i}.nicla_accY./LSB_g_ACC;
        single_experiment_data{i}.nicla_accZ = single_experiment_data{i}.nicla_accZ./LSB_g_ACC;
        
        % Gyroscope conversion (°/s)
        single_experiment_data{i}.nicla_gyroX = single_experiment_data{i}.nicla_gyroX./LSB_deg_s_GYRO;
        single_experiment_data{i}.nicla_gyroY = single_experiment_data{i}.nicla_gyroY./LSB_deg_s_GYRO;
        single_experiment_data{i}.nicla_gyroZ = single_experiment_data{i}.nicla_gyroZ./LSB_deg_s_GYRO;
        
        % Removing useless columns
        single_experiment_data{i}(:,{'nicla_ts','nicla_roll','nicla_pitch','gps_time','gps_spd'})=[];
    
        % Converting motor speed from 10*km/h
        single_experiment_data{i}(:,{'motor_spd'})=single_experiment_data{i}(:,{'motor_spd'})./10;
    
        % Timestamp conversion
        single_experiment_data{i}.timestamp = datetime(single_experiment_data{i}.timestamp,'ConvertFrom','posixtime','TimeZone','Europe/Rome');
        % Removing NaN IMU measurements
        single_experiment_data{i} = single_experiment_data{i}(~isnan(single_experiment_data{i}.nicla_accZ),:);
        
        % Removing low velocity measurements
        single_experiment_data{i}=single_experiment_data{i}(~isnan(single_experiment_data{i}.motor_spd),:);
        single_experiment_data{i} = single_experiment_data{i}(single_experiment_data{i}.motor_spd > mean(single_experiment_data{i}.motor_spd-1),:);

        % Rotating IMU measurements
        single_experiment_data{i}(:,2:4) = array2table((R * table2array(single_experiment_data{i}(:,2:4))')');
        single_experiment_data{i}(:,5:7) = array2table((R * table2array(single_experiment_data{i}(:,5:7))')');

        % Removing gravity from accelerometer's z-axes 
        single_experiment_data{i}.nicla_accZ = single_experiment_data{i}.nicla_accZ +1;

        % Normalizing acceleration by speed
        single_experiment_data{i}.nicla_accX = single_experiment_data{i}.nicla_accX ./ single_experiment_data{i}.motor_spd;
        single_experiment_data{i}.nicla_accY = single_experiment_data{i}.nicla_accY ./single_experiment_data{i}.motor_spd;
        single_experiment_data{i}.nicla_accZ = single_experiment_data{i}.nicla_accZ ./single_experiment_data{i}.motor_spd;

        single_experiment_data{i}.nicla_gyroX = single_experiment_data{i}.nicla_gyroX ./ single_experiment_data{i}.motor_spd;
        single_experiment_data{i}.nicla_gyroY = single_experiment_data{i}.nicla_gyroY ./single_experiment_data{i}.motor_spd;
        single_experiment_data{i}.nicla_gyroZ = single_experiment_data{i}.nicla_gyroZ ./single_experiment_data{i}.motor_spd;

        % Computing norm of accelerations
        single_experiment_data{i}.norm_acc = sqrt(single_experiment_data{i}.nicla_accX.^2 + ...
            single_experiment_data{i}.nicla_accY.^2+ single_experiment_data{i}.nicla_accZ.^2);
    end
end

clear("n_experiment");

%% Different assistance level
% Looking for differences in FFT as the level of assistance varies

terrain = 2; % terrain selection (1->asphalt, 2->gravel) (not enough data for cobblestone)
feature = 'nicla_gyroY';
velocity = 0; % 0 -> 15km/h, 3 -> 25km/h

[sensor, vel_txt, ~ , UoM] = get_texts(feature,velocity,0);

figure;
for i=1:3
    single_experiment_data = experiment_picking(divided_experiments_data,velocity+i);
    X = single_experiment_data{terrain}.(feature);
    
    [P1,f] = windowed_fft(X,1000,100);

    hold on
    if ~isnan(P1)
        plot(f,P1)
    end

    title(strcat("FFTs of ",sensor))
    subtitle(strcat(terrains{terrain}, " - ", vel_txt))
    xlabel("Frequency [Hz]")
    ylabel(strcat("Amplitude ",UoM))
    legend(["Level 0", "Level 3" ,"Level 5"])
    %ylim([0 1])
    xlim([0 50])

end
%print(p,"Img/freq_analysis/assistance_levels_25kmh/gyroZ.png","-dpng")

clear ("f"); clear("i"); clear("P1"); clear("X"); 
clear("sensor"); clear("UoM");clear("vel_txt"); clear("velocity");clear("terrain"); clear("feature");

%% Different speeds FFTs
% Looking for differences in FFT as the speed varies
figure;
lvl_assistance = 2;% 0 ->lvl 0, 1 ->lvl 3 , 2->lvl 5
terrain = 2; %cobblestone available only with lvl 5
feature = "nicla_gyroY";

[sensor, ~, lvl_txt , UoM] = get_texts(feature,0,lvl_assistance);

for i=[1+lvl_assistance,4+lvl_assistance]
    single_experiment_data = experiment_picking(divided_experiments_data,i); 
    X = single_experiment_data{terrain}.(feature);

    [P1,f] = windowed_fft(X,1000,100);

    hold on
    plot(f,P1) 
    title(strcat("FFTs of ",sensor))
    subtitle(strcat(terrains{terrain}, " - ", lvl_txt))
    xlabel("Frequency [Hz]")
    ylabel(strcat("Amplitude ",UoM))
    legend(["15km/h", "25km/h"])
    %ylim([0 0.009])
    xlim([0 50])

end
%print(p,"Img/freq_analysis/speeds_lvl5/accZ.png","-dpng")

clear ("f"); clear("i"); clear("P1"); clear("X");
clear("sensor"); clear("UoM");clear("lvl_txt"); clear("lvl_assistance");clear("terrain"); clear("feature");

%% Damper differences FFTs
% Looking for differences in FFT with and without damper
figure;
velocity = 0;% 0 -> 15km/h, 1 -> 25km/h
terrain = 1; 
feature = "nicla_accZ";

[sensor, vel_txt, ~ , UoM] = get_texts(feature,velocity,0);

for i=[3+(velocity*3),7+velocity]
    single_experiment_data = experiment_picking(divided_experiments_data,i); 
    X = single_experiment_data{terrain}.(feature);

    [P1,f] = windowed_fft(X,1000,100);

    hold on
    plot(f,P1) 
    title(strcat("FFTs of ",sensor))
    subtitle(strcat(terrains{terrain}, " - ", vel_txt))
    xlabel("Frequency [Hz]")
    ylabel(strcat("Amplitude ",UoM))
    legend(["No damper", "Damper"])
    %ylim([0 0.009])
    xlim([0 50])

end
%print(p,"Img/freq_analysis/damper/accZ.png","-dpng")

clear ("f"); clear("i"); clear("P1"); clear("X");
clear("sensor"); clear("UoM");clear("vel_txt"); clear("velocity");clear("terrain"); clear("feature");
%% Different terrains FFT
% Looking for differences in FFT in different terrains
lvl_assistance = 2;% 0 ->lvl 0, 1 ->lvl 3 , 2->lvl 5
velocity = 4; % 1 -> 15km/h , 4 -> 25km/h
feature = "nicla_accX";

[sensor, vel_txt, lvl_txt , UoM] = get_texts(feature,velocity,lvl_assistance);

single_experiment_data = experiment_picking(divided_experiments_data,velocity+lvl_assistance); 
figure;
for i=1:length(single_experiment_data)
    X = single_experiment_data{i}.(feature);

    [P1,f] = windowed_fft(X,1000,100);

    hold on
    plot(f,P1) 
    title(strcat("FFTs of ", sensor))
    subtitle(strcat(vel_txt," - ",lvl_txt))
    xlabel("Frequency [Hz]")
    ylabel(strcat("Amplitude ",UoM))
    legend(terrains)
    %ylim([0 0.007])
    xlim([0 50])

end
%print(p,"Img/freq_analysis/terrain_differences_25kmh/gyroZ.png","-dpng")

clear ("f"); clear("i"); clear("P1"); clear("X");
clear("sensor"); clear("UoM");clear("lvl_txt"); clear("lvl_assistance");clear("terrain"); clear("feature"); clear("velocity"); clear("vel_txt");

%% Feature selection (terrain - velocity differences)
% differences with velocity changes and damper
% due to scaling, 25km/h FFTs have lower amplitude
% due to damper effect, Damper FFTs are lower than no dumper FFTs
% lower case: 25km/h dumper
% highest case: 15km/h no dumper
feature = "nicla_accX";

[sensor, ~, ~ , UoM] = get_texts(feature,0,0);

figure;
for j = [1,8] %1 = 15km/h no dumper, 8 = 25km/h dumper
    single_experiment_data = experiment_picking(divided_experiments_data,j); % 

    for i=1:length(single_experiment_data)
        X = single_experiment_data{i}.(feature);
        
        [P1,f] = windowed_fft(X,1000,100);

        linestyle ="";

        if(j>3)
            linestyle = "--";
        end
    
        colors = ["#0072BD";"#D95319";"#EDB120";];
        hold on
        plot(f,P1,linestyle) 
        title(strcat("FFTs of ", sensor))
        xlabel("Frequency [Hz]")
        ylabel(strcat("Amplitude ",UoM))
        legend(["Asphalt - 15km/h - No damper","Gravel - 15km/h - No damper", "Cobblestone - 15km/h - No damper" ...
            ,"Asphalt - 25km/h - Damper","Gravel - 25km/h - Damper", "Cobblestone - 25km/h - Damper"])
        colororder(colors)

        ylim([0 1.2])
        xlim([0 50])

        if(j>3)
            patch([lines{i}{1} fliplr(lines{i}{1})], [lines{i}{2} fliplr(P1)], hex2rgb(colors(i)),'HandleVisibility','off','FaceAlpha',.3)
        end
    
        lines{i} = {f, P1};
    end
end
%print(p,"Img/freq_analysis/terrain_differences_25kmh/gyroZ.png","-dpng")

clear ("f"); clear("i"); clear("P1"); clear("X");
clear("sensor"); clear("UoM"); clear("feature"); clear("j"); clear("linestyle"); clear("colors"); clear("lines");
%% Get title/subtitle/label function

function [sensor_name, vel_txt, assistance_lvl_txt, label_UoM] = get_texts(feature, velocity, assistance_lvl)
    
    %Text for sensor
    if(feature == "nicla_accX")
        sensor_name = "accelerometer (X-Axis)";
    elseif(feature == "nicla_accZ")
        sensor_name = "accelerometer (Z-Axis)";
    elseif(feature == "nicla_gyroY")
        sensor_name = "gyroscope (Y-Axis)";
    else
        sensor_name = "";
    end

    %Text for velocity
    if(velocity==0)
        vel_txt="15km/h";
    else
        vel_txt ="25km/h";
    end

    %Text for assistance lvl 
    if(assistance_lvl==0)
        assistance_lvl_txt="Assistance level 0";
    elseif(assistance_lvl ==1)
        assistance_lvl_txt ="Assistance level 3";
    else
        assistance_lvl_txt = "Assistance level: 5";
    end
    %Text for UoM
    if (contains(feature,'acc'))
        label_UoM = "[g]";
    else
        label_UoM = "[°/s]";
    end

end