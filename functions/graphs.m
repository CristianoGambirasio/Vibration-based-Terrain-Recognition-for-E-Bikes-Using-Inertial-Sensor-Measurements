function [] = graphs(terrains,data,imu,enviroment,position, motor)
%GRAPHS
%   Display graphs for the dataset
%
% INPUTS
% 
% terrains = 
%   list of terrains names. Ex: {'terr1', 'terr2', ...}
% 
% data = 
%   list of tables containing data for every terrain. Ex: {data_terr1, data_terr2, ...}
% 
% imu,enviroment,position,motor = 
%   booleans to select which graphs to show

for i=1:length(terrains)
    if (isnumeric(data{i}.timestamp) || isreal(data{i}.timestamp)) % Check if timestamp is already converted
      data{i}.timestamp = datetime(data{i}.timestamp,'ConvertFrom','posixtime','TimeZone','Europe/Rome');
    end
    experiment_lines{i} = find(diff(data{i}.timestamp) > minutes(1) | diff(data{i}.timestamp) < minutes(-1));
    experiment_lines{i} = [0; experiment_lines{i}];
end

%% IMU visualization
% Two figures,
% 1 - 3 axis of accelerometer and gyroscope
% 2 - norm of accelerometer
if(imu)
    for i=1:length(terrains)
        figure('NumberTitle', 'off', 'Name', strcat(terrains{i},' IMU Visualization'));
        subplot(3,2,1)
        plot(data{i}.nicla_accX)
        title('Accelerometer')
        subtitle('X-axis')
        %xlabel('Number of sample')
        %ylabel('Acceleration [g]')
        ylim([-0.2, 0.2])
        xline(experiment_lines{i},'r','LineWidth',1,'DisplayName',"Experiments division")
        legend('','Experiments divider');
        
        subplot(3,2,3)
        plot(data{i}.nicla_accY)
        subtitle('Y-axis')
        %xlabel('Number of sample')
        ylabel('Acceleration [g]')
        ylim([-0.05, 0.05])
        xline(experiment_lines{i},'r','LineWidth',1)

        subplot(3,2,5)
        plot(data{i}.nicla_accZ)
        subtitle('Z-axis')
        xlabel('Number of sample')
        %ylabel('Acceleration [g]')
        ylim([-0.2, 0.2])
        xline(experiment_lines{i},'r','LineWidth',1)

        subplot(3,2,2)
        plot(data{i}.nicla_gyroX)
        title('Gyroscope')
        subtitle('X-axis')
        %xlabel('Number of sample')
        %ylabel('Velocity [°/s]')
        ylim([-3, 3])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,4)
        plot(data{i}.nicla_gyroY)
        subtitle('Y-axis')
        %xlabel('Number of sample')
        ylabel('Velocity [°/s]')
        ylim([-2, 2])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,6)
        plot(data{i}.nicla_gyroZ)
        subtitle('Z-axis')
        xlabel('Number of sample')
        %ylabel('Velocity [°/s]')
        ylim([-5, 5])
        xline(experiment_lines{i},'r','LineWidth',1)

        figure('NumberTitle', 'off', 'Name', strcat(terrains{i},' Norm Visualization'));
        plot(data{i}.norm_acc)
        title("Norm of acceleration")
        xlabel("Number of sample")
        ylabel("Acceleration [g]")
        ylim([0, 0.3])
        xline(experiment_lines{i},'r','LineWidth',1,'DisplayName',"Experiments division")
        legend('','Experiments divider');

    end
end

%% Enviroment data visualization
% figure of enviromental features
if(enviroment)
    for i=1:length(terrains)
    
        f = figure('NumberTitle', 'off', 'Name', strcat(terrains{i},' Enviroment Data Visualization'));
        subplot(3,2,1)
        plot(data{i}.nicla_temp)
        title('Enviromental data')
        subtitle('Temperature')
        xlabel('Number of sample')
        ylabel('Temperature [°C]')
        ylim([25, 40])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,3)
        plot(data{i}.nicla_hum)
        subtitle('Humidity')
        xlabel('Number of sample')
        ylabel('Humidity [%]')
        ylim([20, 60])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,5)
        plot(data{i}.nicla_pres)
        subtitle('Pressure')
        xlabel('Number of sample')
        ylabel('Pressure [Pa]')
        ylim([9.93* 10^4, 9.95* 10^4])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,2)
        plot(data{i}.nicla_iaq)
        title('Air quality')
        subtitle('Air quality index')
        xlabel('Number of sample')
        ylabel('Quality index')
        ylim([0, 200])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,[4,6])
        plot(data{i}.nicla_gas)
        subtitle('MOx sensor')
        xlabel('Number of sample')
        ylabel('Resistance [Ω]')
        ylim([1 * 10^4, 3.5* 10^4])
        xline(experiment_lines{i},'r','LineWidth',1)
    end
end

%% Position visualization
% map of gps measurements
if(position)
    
    for i=1:length(terrains)
        figure;
        geobasemap satellite
        geoplot(data{i}.gps_lat(1:50000),data{i}.gps_lon(1:50000),"LineWidth",3)
    end
end

%% Motor data visualization
% Figure of motor data measurements
if(motor)
    for i=1:length(terrains)
    
        figure('NumberTitle', 'off', 'Name', strcat(terrains{i},' Motor data visualization'));
        subplot(3,2,1)
        yyaxis left
        plot(data{i}.motor_soc)
        ylabel('Battery [%]')
        ylim([0, 100])
        yyaxis right
        plot(data{i}.motor_curr_cap)
        ylabel('Battery [Ah]')
        title('Battery data')
        subtitle('Remaining battery')
        xlabel('Number of sample')
        ylim([0, 20])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        
        subplot(3,2,3)
        plot(data{i}.motor_res_range)
        subtitle('Residual Range')
        xlabel('Number of sample')
        ylabel('Distance available [m]')
        ylim([0, 150000])
        xline(experiment_lines{i},'r','LineWidth',1)
        
        subplot(3,2,2)
        yyaxis left
        plot(data{i}.motor_curr_level)
        ylabel('Assistance level')
        ylim([0 5.1])
        yyaxis right
        plot(data{i}.motor_assistance, Color=[0.4660 0.6740 0.1880])
        ylabel('Assistance perc. [%]')
        title('Motor data')
        subtitle('Assistance')
        xlabel('Number of sample')
        ylim([0, 100])
        xline(experiment_lines{i},'r','LineWidth',1)
        ax = gca;
        ax.YAxis(2).Color = [0.4660 0.6740 0.1880];
        
        subplot(3,2,4)
        yyaxis left
        plot(data{i}.motor_rider_pw)
        ylabel('Rider power [W]')
        ylim([0 500])
        yyaxis right
        plot(data{i}.motor_cadence, Color=[0.4660 0.6740 0.1880])
        ylabel('Motor cadence [rpm]')
        subtitle('Rider power')
        xlabel('Number of sample')
        ylim([0 100])
        xline(experiment_lines{i},'r','LineWidth',1)
        ax = gca;
        ax.YAxis(2).Color = [0.4660 0.6740 0.1880];
         
        subplot(3,2,[5,6])
        plot(data{i}.motor_spd)
        ylabel('Speed [km/h]')
        subtitle('Motor speed')
        xlabel('Number of sample')
        %ylim([0, 30])
        xline(experiment_lines{i},'r','LineWidth',1)
    end
    
end



end

