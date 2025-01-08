%WINDOWED_FFT
%   compute FFT on window with selected length and returns the mean result
%
% INPUTS
% 
% X = 
%   sequence of measurements
%
% window_length = 
%   length of the window  
%   
% overlap = 
%   length of overlapping segment between two windows
%
% OUTPUTS
% 
% ffts_mean = 
%   mean of the computed ffts
% 
% f = 
%   corrisponding frequencies for 'ffts_mean'

function [ffts_mean,f] = windowed_fft(X,window_length, overlap)
    N = length(X); 
    Fs = 100; % Sampling frequency

    hann_window = hann(window_length); %hanning the window
    
    num_segments = floor((N - overlap) / (window_length - overlap));

    fft_segments = [];

    % Dividing signal
    for j = 1:num_segments
        start_idx = (j - 1) * (window_length - overlap) + 1;
        end_idx = start_idx + window_length - 1;
        if end_idx > N
            break;
        end

        segment = X(start_idx:end_idx);
        windowed_segment = segment .* hann_window;

        fft_segment = fft(windowed_segment); % computing FFT for segment
        
        fft_segment = fft_segment(1:floor(window_length/2))';

        fft_segments = [fft_segments; abs(fft_segment)];
    end

    ffts_mean = mean(fft_segments); % means of ffts
    f = (0:floor(window_length/2)-1) * Fs / window_length;
end

