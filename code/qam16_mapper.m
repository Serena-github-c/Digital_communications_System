function symbols = qam16_mapper(bitstream)
  % QAM16_MAPPER Map a binary bitstream into Gray-coded 16-QAM symbols
  %   symbols = qam16_mapper(bitstream)
  %
  % Input:
  %   bitstream : row‐vector of 0/1 values, length must be multiple of 4
  %
  % Output:
  %   symbols   : complex row‐vector of 16-QAM constellation points

  % reshape into N×4 matrix
  bits = reshape(bitstream(:), 4, []).';

  % convert each 4-bit group to decimal [0..15], Gray‐coded bit order (MSB first)
  idx = bi2de(bits, 'left-msb');

  % Gray-coded mapping table for 16-QAM
  % Rows 1..16 correspond to idx = 0..15
  mapping_table = [
    -3 -3;  -3 -1;  -3 +3;  -3 +1;
    -1 -3;  -1 -1;  -1 +3;  -1 +1;
    +3 -3;  +3 -1;  +3 +3;  +3 +1;
    +1 -3;  +1 -1;  +1 +3;  +1 +1
  ];

  % pick rows (add 1 since Octave indices start at 1)
  pts = mapping_table(idx+1, :);

  % form complex symbols
  symbols = pts(:,1).' + 1i*pts(:,2).';
end

