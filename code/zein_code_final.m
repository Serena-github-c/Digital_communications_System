pkg load signal
pkg load communications

%% Step 1: Random Analog Signal Generation
Fs = 2000;              % Sampling frequency (Hz)
T = 1;                   % Duration (seconds)
t = 0:1/Fs:T-1/Fs;       % Time vector
rand("seed", 123);

%% 1.1 Generate an Analog-Like Signal (Sum of Sinusoids)
frequencies = [30, 80, 150, 250, 350];  % Hz
amplitudes  = [1.0, 0.7, 0.5, 0.4, 0.3];     % Amplitudes
phases = 2*pi*rand(1, length(frequencies));  % Random phases

x = zeros(size(t));  % Initialize signal
for i = 1:length(frequencies)
    x = x + amplitudes(i) * sin(2*pi*frequencies(i)*t + phases(i));
end

% Normalize signal to [-1, 1]
x = x / max(abs(x));
sound(x,Fs);
% Write signal to an audio file
audiowrite('generated_signal.wav', x, Fs);


%% 1.2 Plot Time-Domain Signal (Continuous)
figure('Name', 'Time-Domain Signal (Analog Continuous)');
plot(t, x, 'b');
xlabel('Time (s)');
ylabel('Amplitude');
title('Time-Domain Signal (Analog Continuous)');
grid on;


%% Frequency-Domain Analysis using FFT
N = length(x);
X_f = abs(fftshift(fft(x))) / N;
f_axis = linspace(-Fs/2, Fs/2, N);


figure('Name', 'Frequency Domain');
plot(f_axis, X_f, 'r');
xlabel('Frequency (Hz)');
ylabel('|X(f)|');
title('Frequency Spectrum of Signal');
grid on;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 2: Random Analog Signal Generation
%% 2.1 Sampling(Continuous-Time to Discrete-Time)
sample_indices = 1:200;  % Display first 200 samples
figure('Name', 'Sampled Signal Representation (Discrete-Time)', 'NumberTitle', 'off');
subplot(3,1,1);
stem(t(sample_indices), x(sample_indices), 'filled'); %filled: dots are filled
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampled Signal Representation (200 samples)');
grid on;

subplot(3,1,2);
stem(t(1:300), x(1:300), 'filled');
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampled Signal Representation (300 samples)');
grid on;

subplot(3,1,3);
stem(t(1:100), x(1:100), 'filled');
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampled Signal Representation (100 samples)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.2 Mid-Rise Quantization with different L values
L_values = [4,8,16];       % Number of quantization levels

for i = 1:length(L_values)
  L = L_values(i);

  xmax = max(abs(x));                  % Maximum amplitude for range
  delta = 2 * xmax / L;                % Step size (Δ)
  q_midrise = delta * (floor(x / delta) + 0.5); % Mid-rise quantization
  error_midrise = x - q_midrise;       % Quantization error

  %% Plottingg: create a new figure for each L
  figure('Name', ['Mid-Rise Quantization - L = ' num2str(L)], 'NumberTitle', 'off');
  subplot(3,1,1);
  plot(t, x, 'b', t, q_midrise, 'r--');
  title(['Original vs Mid-Rise Quantized Signal (L = ' num2str(L) ')']);
  xlabel('Time (s)');
  ylabel('Amplitude');
  legend('Original', 'Quantized');
  grid on;

  subplot(3,1,2);
  plot(t, error_midrise, 'k');
  title('Quantization Error');
  xlabel('Time (s)');
  ylabel('Error');
  grid on;

  subplot(3,1,3);
  stairs(t, q_midrise, 'r');
  title('Quantized Signal Levels (Staircase Plot)');
  xlabel('Time (s)');
  ylabel('Quantized Levels');
  grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%Step 3: PCM Encoding
  q_index = floor((q_midrise + xmax) / delta);  % map to integers
  n_bits = log2(L);
  pcm_binary = dec2bin(q_index, n_bits);  % matrix of bits
  bitstream = reshape(pcm_binary.' - '0', 1, []);  % flat binary stream

  disp(['First 8 PCM binary codes (L = ' num2str(L) '):']);
  disp(pcm_binary(1:8,:));

  % Save bitstream for L=16 to use later
  if L == 16
    bitstream_L16 = bitstream;
  endif


  % Visualize PCM bitstream
  figure('Name', ['PCM Bitstream - L = ' num2str(L)], 'NumberTitle', 'off');
  stem(bitstream(1:200), 'filled');
  title(['PCM Bitstream (First 200 bits) for L = ' num2str(L)]);
  xlabel('Bit Index');
  ylabel('Bit Value');
  ylim([-0.1 1.1]);
  grid on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 4: Baseband Modulation
%% 4.1 Line Coding – Bipolar NRZ(Binary -> Bipolar waveform)

% Parameters : Continue with L=16
L = 16;
bitrate = Fs * log2(L);   % bps
Tb = 1 / bitrate;         % duration of 1 bit

samples_per_bit = 20;    % Oversampling factor for smooth plot, each digital bit is stretched to 20 points in time
bitstream = double(bitstream_L16);  % Use the saved bitstream from L = 16
n_bits = length(bitstream);
disp(bitstream(1:30));


% Time vector for waveform (each bit is expanded over Tb)
t_line = 0:Tb/samples_per_bit:(n_bits * Tb - Tb/samples_per_bit);


% NRZ signal generation
nrz_waveform = zeros(1, length(t_line));

for b = 1:n_bits
    if bitstream(b) == 1
      bit_val = +1;
    else
      bit_val = -1;
    end
    start_idx = (b-1)*samples_per_bit + 1;
    end_idx = b*samples_per_bit;
    nrz_waveform(start_idx:end_idx) = bit_val;
end


% for debugging
disp('First 30 bits in the bitstream:');
disp(bitstream(1:30));
disp(nrz_waveform(1:30));  % should be mostly blocks of +1 or -1, not jumping


% Plot NRZ waveform: first 30 bits to see them clearly
N_plot_bits = 30;
plot_idx = 1 : N_plot_bits * samples_per_bit;


figure('Name', 'NRZ Line Coding - L = 16 -  First 30 bits', 'NumberTitle', 'off');
plot(t_line(plot_idx), nrz_waveform(plot_idx), 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Amplitude');
title(['NRZ Line Coding of Bitstream (L = 16)']);
ylim([-1.5, 1.5]);
xlim([t_line(plot_idx(1)), t_line(plot_idx(end))]);
grid on;
hold on;

% Add vertical lines to mark each bit boundary
for b = 1:N_plot_bits+1
    x = t_line((b-1)*samples_per_bit+1);
    line([x x], [-1.5 1.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');  % Light gray dashed lineend
end

% To label each bit
for b = 1:N_plot_bits
    bit_value = bitstream(b);
    bit_center = t_line((b-1)*samples_per_bit + round(samples_per_bit/2));

    y_pos = 1.3 * (2 * bit_value - 1);  % +1.3 or -1.3
    text(bit_center, y_pos, num2str(bit_value), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'Color', 'b');
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Raised cosine filtering after NRZ

beta = 0.5;
span = 6;
sps = 20;

% Get Raised Cosine filter coefficients
h_rc = raised_cosine(beta, span, sps);

% Apply the filter to the NRZ signal
nrz_shaped = conv(nrz_waveform, h_rc, 'same');

% Create a time vector for plotting
t_shaped = linspace(0, length(nrz_shaped) / (Fs * log2(L)), length(nrz_shaped));

% === Plot ONLY the first 30 bits ===
N_plot_bits = 30;
samples_per_bit = 20;
sps = samples_per_bit;  % same thing

% Calculate how many samples that means:
plot_samples = N_plot_bits * samples_per_bit;

% Create time vector for those samples
t_shaped_zoom = linspace(0, N_plot_bits * Tb, plot_samples);

% Extract the first part of the shaped signal
nrz_shaped_zoom = nrz_shaped(1:plot_samples);

% Plot
figure('Name', 'Raised Cosine Filtered Signal - First 30 Bits', 'NumberTitle', 'off');
plot(t_shaped_zoom, nrz_shaped_zoom, 'b', 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Amplitude');
title('Raised Cosine Shaped NRZ - First 30 Bits');
grid on;

% Add vertical lines to show bit boundaries
hold on;
for b = 1:N_plot_bits+1
    x = (b-1)*Tb;
    line([x x], [-1.5 1.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end

% Label the bit values on top of the waveform
for b = 1:N_plot_bits
    bit_value = bitstream(b);
    bit_center_time = (b - 0.5) * Tb;

    % Choose vertical position based on bit value
    if bit_value == 1
        y_pos = 1.2;
    else
        y_pos = -1.2;
    end

    text(bit_center_time, y_pos, num2str(bit_value), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 10, 'Color', 'b');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EYe diagram to see ISI
bits_per_eye = 2;
plot_eye_diagram(nrz_shaped, samples_per_bit, Tb, bits_per_eye, 'Raised Cosine Filtered');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: 16-QAM Modulation

qam_symbols = qam16_mapper(bitstream);

%--- 5.2: Upsample & Pulse-shape ---
sps   = samples_per_bit * 4;
upsym = upsample(qam_symbols, sps);         % from signal pkg
rc    = raised_cosine(0.5, 6, sps);         % your helper .m
qam_shaped = conv(upsym, rc, 'same');

%--- 5.3: Bandpass Modulation ---
fc   = 400;
t    = (0:length(qam_shaped)-1)/Fs;
qam_tx = real(qam_shaped .* exp(1i*2*pi*fc*t));


%--- 5.4: Plots ---
figure('Name', '16-QAM Bandpass', 'NumberTitle', 'off');

plot(t(1:600), qam_tx(1:600));
xlabel('Time (s)'); ylabel('Amplitude');
title('16-QAM Bandpass (first 600 samples)'); grid on;

% spectrum analysis (Fourrier)
Nfft = 2^nextpow2(length(qam_tx));
Q    = fftshift(abs(fft(qam_tx, Nfft)));
f    = linspace(-Fs/2, Fs/2, Nfft);
figure('Name', 'Spectrum of 16-QAM Bandpass', 'NumberTitle', 'off');
plot(f, Q);
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Spectrum of 16-QAM Signal'); grid on;

% Constellation
I = real(qam_shaped(1:sps:end));
Qv= imag(qam_shaped(1:sps:end));
if exist('scatter','file')
  figure('Name', '16-QAM Constellation', 'NumberTitle', 'off');
  scatter(I, Qv, 20, 'filled');
else
  figure('Name', '16-QAM Constellation', 'NumberTitle', 'off');
  plot(I, Qv, 'o', 'MarkerFaceColor','b');
end
xlabel('In-phase'); ylabel('Quadrature');
title('16-QAM Constellation'); axis square; grid on;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 6: Channel Simulation

% 6.1: AWGN
SNR_dB       = 20;
signal_power = mean(abs(qam_tx).^2);
noise_power  = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power/2) * (randn(size(qam_tx)) + 1i*randn(size(qam_tx)));
rx_awgn = qam_tx + noise;

% 6.2: Optional Flat Rayleigh fading
apply_fading = false;

% Always start with the AWGN-only signal
rx = rx_awgn;

if apply_fading
  % generate one Rayleigh tap per symbol
  Nsym = length(qam_symbols);
  h_sym = (randn(1,Nsym) + 1i*randn(1,Nsym)) / sqrt(2);
  % upsample taps to per-sample gains
  h = upsample(h_sym, sps);
  h = h(1:length(rx));    % trim to match length
  rx = rx .* h;           % apply fading
end

% 6.3: Time-domain plot (first 600 samples)
t = (0:length(rx)-1) / Fs;
figure('Name', 'Received Signal (real part) – AWGN + Rayleigh fading', 'NumberTitle', 'off');
plot(t(1:600), real(rx(1:600)), 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Amplitude');
if apply_fading
  title('Received Signal (real part) – AWGN + Rayleigh fading');
else
  title('Received Signal (real part) – AWGN only');
end
grid on;

% 6.4: Frequency-domain plot
Nfft = 2^nextpow2(length(rx));
R    = fftshift(abs(fft(rx, Nfft)));
f    = linspace(-Fs/2, Fs/2, Nfft);
figure('Name', 'Spectrum of Received Signal', 'NumberTitle', 'off');
plot(f, R, 'LineWidth', 1.2);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Spectrum of Received Signal');
grid on;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 7: Reciever side

% 7.1 Coherent demodulation
t = (0:length(rx)-1)/Fs;
baseband = rx .* exp(-1i*2*pi*fc*t);

figure('Name', 'Coherently Demodulated Baseband ', 'NumberTitle', 'off');
plot(t(1:600), real(baseband(1:600)), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Coherently Demodulated Baseband (first 600 samples)');
grid on;


% 7.2 Matched filter
mf_output = conv(baseband, rc, 'same');
bits_per_eye = 2;
plot_eye_diagram(mf_output, sps, Tb, bits_per_eye, 'MF Output Eye Diagram');


% 7.3 Symbol sampling & detection
delay = span*sps/2;
start = delay + 1;
% compute how many symbols actually fit without running past mf_output
max_syms = floor((length(mf_output) - start) / sps) + 1;

pos = start : sps : start + (max_syms-1)*sps;
rx_samps = mf_output(pos);

% redefine the mapping table
mapping_table = [
    -3 -3;  -3 -1;  -3 +3;  -3 +1;
    -1 -3;  -1 -1;  -1 +3;  -1 +1;
    +3 -3;  +3 -1;  +3 +3;  +3 +1;
    +1 -3;  +1 -1;  +1 +3;  +1 +1
  ];


pts = mapping_table(:,1) + 1i*mapping_table(:,2);
det_idx = arrayfun(@(x) find(min(abs(x - pts))==abs(x - pts),1), rx_samps);

figure('Name', 'Constellation of Sampled Symbols', 'NumberTitle', 'off');
scatter(real(rx_samps), imag(rx_samps), 20, 'filled');
xlabel('In-phase'); ylabel('Quadrature');
title('Constellation of Sampled Symbols');
axis square; grid on;


% 7.4 Bit recovery
dec = det_idx - 1;
bits4 = de2bi(dec, 4, 'left-msb');
recovered_stream = reshape(bits4.', 1, []);

%% --- Plot the original analog-like signal ---
t = 0:1/Fs:T-1/Fs;    % recreate the original time vector
figure('Name', 'Original Generated Analog-Like Signal', 'NumberTitle', 'off');
plot(t, x, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Generated Analog-Like Signal');
grid on;

%% --- Normalize & write the received signal to WAV ---
y = real(rx(1:2*Fs));
y = y / max(abs(y));           % scale to full span
audiowrite('received_signal.wav', y, Fs);

sound(y,Fs);



