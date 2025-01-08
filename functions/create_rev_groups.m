function [outData] = create_rev_groups(inData,R_WHEEL,N_REV_GROUP)
%CREATE_REV_GROUPS
%   Adds a column to input data for the division of dataset into groups of
%   N_REV_GROUP wheel revolutions
% 
% INPUTS
% inData=
%     list of tables containing data for every terrain. Ex: {data_terr1, data_terr2, ...}
% R_WHEEL=
%     radius of wheel in meter
% 
% N_REV_GROUP=
%     Number of revolutions per group
%
% OUTPUT
% outData =
%       inData with "group" column

% Input checking
if(istable(inData))
    inData = {inData};
end

% Grouping algorithm
for i=1:length(inData)
    data_speed = inData{i}(~isnan(inData{i}.motor_spd),:);% Remove NaN motor speed
    data_speed = data_speed(data_speed.motor_spd~= 0,:); % Remove 0 motor speed
    
    start_idx = 1;
    group = 1;
    
    data_speed.group = zeros(height(data_speed),1);
    
    while start_idx<=height(data_speed)
        % Computing rev/s at current velocity
        rev_s = (data_speed(start_idx,:).motor_spd/3.6)*(1/(2*pi*R_WHEEL)); % /3.6 because speed is in km/h
        
        % Computing time to complete N_REV_GROUP revolutions and group
        % dimension
        group_dim = floor((N_REV_GROUP/rev_s) * 100); %sample every 10ms
        
        end_idx = start_idx + group_dim; % group end index
        
        % Assigning groups
        if end_idx<= height(data_speed)
            data_speed.group(start_idx:end_idx) = group;
        else %last group handling
            if(height(data_speed(start_idx:end,:))<10) % Avoid groups with less than 10 measurements
                data_speed.group(start_idx:end) = group-1;
            else  
                data_speed.group(start_idx:end) = group;
            end
        end
    
        group = group+1;
        start_idx = end_idx+1;
    end
    outData{i} = data_speed;
end

% Return table if input is a table
if isscalar(outData)
    outData = outData{1};
end

end

