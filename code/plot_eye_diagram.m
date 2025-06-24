function plot_eye_diagram(signal, samples_per_bit, Tb, bits_per_eye, title_str)
  % signal: the shaped signal (e.g., NRZ or raised cosine)
  % samples_per_bit: number of samples per bit
  % Tb: bit duration (1/bitrate)
  % bits_per_eye: number of bits per segment (usually 2)
  % title_str: string for the plot title

  samples_per_eye = bits_per_eye * samples_per_bit;
  max_eyes = 100;  % Limit to avoid overload
  num_eyes_total = floor(length(signal) / samples_per_eye);
  num_eyes = min(num_eyes_total, max_eyes);
  eye_matrix = zeros(num_eyes, samples_per_eye);

  for i = 1:num_eyes
      idx_start = (i - 1) * samples_per_eye + 1;
      idx_end   = idx_start + samples_per_eye - 1;
      eye_matrix(i, :) = signal(idx_start:idx_end);
  end

  t_eye = linspace(0, bits_per_eye * Tb, samples_per_eye);

  figure('Name', 'Eye Diagram', 'NumberTitle', 'off');
  plot(t_eye, eye_matrix', 'b');
  xlabel('Time (s)');
  ylabel('Amplitude');
  title(['Eye Diagram - ' title_str]);
  grid on;
end

