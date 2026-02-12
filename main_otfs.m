clc;
clear;
close all;

%Signal → OTFS modulation → wireless channel → receiver → performance metric

M=16;  % delay bins, multipath
N=16;  % doppler bins, mobility  
%256 symbols

bits=randi([0,1], M*N*2, 1);    %2bits - 1 symbol in QAM

real_bits=bits(1:2:end);
imag_bits=bits(2:2:end);

real_part=2*real_bits - 1;
imag_part=2*imag_bits - 1;

symbols=real_part + 1j*imag_part;       %bits to constellation 

disp("First 10 bits: ");
disp(bits(1:10));

disp("First 5 Symbols: ");
disp(symbols(1:5));

figure;
plot(real(symbols),imag(symbols),'o');
title('QPSK Constellation');
grid on;
axis equal;

X_dd = zeros(M,N);

pilot_amp = 5;     % strong pilot
pilot_row = M/2;
pilot_col = N/2;

X_dd(pilot_row, pilot_col) = pilot_amp;

% Fill remaining cells with data
mask = true(M,N);
mask(pilot_row, pilot_col) = false;

data_symbols = symbols(1:sum(mask(:)));
X_dd(mask) = data_symbols;
% Modified delay doppler domain

figure;
imagesc(abs(X_dd));
title('Pilot + Data DD Grid');
colorbar;

disp("Sample DD grid values: ");
disp(X_dd(1:3,1:3));

%radio transmits signal in time freq domain not in DD domain

X_tf=fft(ifft(X_dd,[],1),[],2);     %delay doppler - time freq
%ifft across delay
%fft across doppler
% spreads the symbol across both time & freq domain

disp("Sample TF grid:");
disp(X_tf(1:3,1:3));

figure;
imagesc(abs(X_tf));
title('Time-Frequency Magnitude');
colorbar;

%heisenberg transform

tx_time=ifft(X_tf,[],1);
tx_signal=tx_time(:);       %time freq to time waveform

disp("First 10 transmit samples: ");
disp(tx_signal(1:10));

SNR_dB=10;          %Adding Noise here onwards
signal_power = mean(abs(tx_signal).^2);
noise_power = signal_power/(10^(SNR_dB/10));

real_noise=randn(size(tx_signal));
imag_noise=randn(size(tx_signal));

real_noise=sqrt(noise_power/2) * real_noise;
imag_noise=sqrt(noise_power/2) * imag_noise;

noise=real_noise+1j*imag_noise;

fd = 50;          % Doppler frequency (try 50–500)
fs = 10000;        % sampling rate (arbitrary simulation rate)

t = (0:length(tx_signal)-1)'/fs;

% multipath parameters
delays  = [0 3 7];
gains   = [1 0.6 0.3];
doppler = [0 40 -30];   % Hz per path

rx_signal = zeros(size(tx_signal));

for k = 1:length(delays)

    d = delays(k);
    g = gains(k);
    fd = doppler(k);

    t = (0:length(tx_signal)-1)'/fs;

    doppler_phase = exp(1j*2*pi*fd*t);

    delayed = [zeros(d,1); tx_signal(1:end-d)];

    rx_signal = rx_signal + g * delayed .* doppler_phase;

end

% add noise
rx_signal = rx_signal + noise;
      %Doppler Channel
        
% r(t)=s(t).e^j*2*pi*Fd*t

rx_time = reshape(rx_signal,M,N);

Y_tf = fft(rx_time,[],1);
Y_dd = fft(ifft(Y_tf,[],2),[],1);

%pilot_rx = Y_dd(pilot_row, pilot_col);

%phi = angle(pilot_rx);   % phase rotation caused by Doppler

%Tframe = length(tx_signal)/fs;

%fd_est = phi/(2*pi*Tframe);          %Estimate Doppler Freq

%doppler_corr = exp(-1j*2*pi*fd*t);

%rx_signal_corr = rx_signal .* doppler_corr;
     %un-rotated signal

rx_time = reshape(rx_signal,M,N);       %to matrix form

Y_tf = fft(rx_time,[],1);                %time to freq

Y_dd = fft(ifft(Y_tf,[],2),[],1);       %reverse isfft

% ----- SIMPLE DD EQUALIZER -----

H_est = Y_dd / pilot_amp;

epsilon = 1e-3;

kernel = ones(3)/9;     % 3×3 smoothing

H_smooth = conv2(H_est, kernel, 'same');

Y_eq = Y_dd;

% ---------

%H_fft = fft2(H_est);

%Y_fft = fft2(Y_data_dd);

%X_fft = Y_fft ./ (H_fft + epsilon);

%Y_eq = ifft2(X_fft);           %Stabilize inversion

%channel_est = Y_dd(pilot_row, pilot_col) / pilot_amp;       % What pilot became/ what pilot was 

%Y_eq = Y_dd / abs(channel_est);
      % Equalising entire grid

%We came back to delay doppler domain

rx_symbols_eq = Y_eq(mask);         %flatten equalised symbols

disp("RX time grid sample:");
disp(rx_time(1:3,1:3));

disp(norm(Y_dd - X_dd));

figure;
imagesc(abs(Y_dd));
title('DD Grid — Multipath + Doppler');
colorbar;

figure;
plot(real(Y_dd(:)), imag(Y_dd(:)),'.');
title('Constellation BEFORE EQ');
axis equal;
grid on;

fprintf("Channel energy estimate = %.3f\n", norm(H_est));

clean_symbols = symbols;        %original
noisy_symbols = rx_symbols_eq;     %distorted receiving 

figure;
plot(real(rx_symbols_eq), imag(rx_symbols_eq),'.');
title('Constellation AFTER EQ');
axis equal;
grid on;

% Extract received data symbols (exclude pilot)
rx_data = Y_eq(mask);

% Detect
detected_data = sign(real(rx_data)) + ...
                1j*sign(imag(rx_data));

% Compare against true transmitted data
errors = sum(detected_data ~= data_symbols);

BER = errors / length(data_symbols);

fprintf("Errors after EQ: %d\n", errors);
fprintf("BER: %.5f\n", BER);

fprintf("Symbol errors: %d\n", errors);

figure;
plot(real(detected_data), imag(detected_data),'o');
title('Detected Symbols');
axis equal;
grid on;

SNR_range = 0:2:20;
BER_curve = zeros(size(SNR_range));

for k = 1:length(SNR_range)

    SNR_dB = SNR_range(k);

    % --- noise generation ---
    signal_power = mean(abs(tx_signal).^2);
    noise_power = signal_power/(10^(SNR_dB/10));

    noise = sqrt(noise_power/2) * ...
        (randn(size(tx_signal)) + 1j*randn(size(tx_signal)));

    % --- channel ---
    rx_signal = zeros(size(tx_signal));

    for p = 1:length(delays)

        d = delays(p);
        g = gains(p);
        fd = doppler(p);

        t = (0:length(tx_signal)-1)'/fs;

        doppler_phase = exp(1j*2*pi*fd*t);

        delayed = [zeros(d,1); tx_signal(1:end-d)];

        rx_signal = rx_signal + g * delayed .* doppler_phase;

    end

    rx_signal = rx_signal + noise;

    % --- OTFS demod ---
    rx_time = reshape(rx_signal,M,N);

    Y_tf = fft(rx_time,[],1);
    Y_dd = fft(ifft(Y_tf,[],2),[],1);

    % --- simple detection ---
    rx_data = Y_dd(mask);

    detected = sign(real(rx_data)) + ...
               1j*sign(imag(rx_data));

    errors = sum(detected ~= data_symbols);

    BER_curve(k) = errors / length(data_symbols);

end

figure;
semilogy(SNR_range, BER_curve,'-o');
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('OTFS BER vs SNR');

figure;     %Clean Grid
subplot(1,2,1);
imagesc(abs(X_dd));
title('Original DD');

subplot(1,2,2);     %received Grid
imagesc(abs(Y_dd));
title('Received DD Grid');

%At time of transmission every symbol had same energy
%While receiving due to noise, the energy of symbol varies, hence change in
%color of grid