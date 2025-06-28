# Digital_communications_System
This project implements an end-to-end digital communication system in Octave GNU (MATLAB).

![image](https://github.com/user-attachments/assets/a24a5c52-6f65-44ee-9234-631aa06a3bcc)
<br><br>
The objective is to understand and visualize each component of the digital communication chain.

The simulation begins with creating a continuous-time, band-limited signal, and then proceeds through digitization, modulation, simulated transmission over a wireless-like channel, and eventual reconstruction of the signal at the receiver side

## Features
- Random analog signal generation and spectral analysis : A sum of random sinusoidals waves converted to audio
- Sampling and uniform quantization (comparing output at different levels)
- PCM encoding and line coding (Ploar NRZ)  
- Pulse shaping using a Raised Cosine filter  (to eliminate ISI) 
- Bandpass modulation (16-QAM)  
- AWGN channel simulation
- Receiver-side processing: demodulation, matched filtering, symbol detection  
- Signal reconstruction and comparison with the original

## Repository Structure
- code : contains the main code implemented in Octave GNU. The main script is 'project_code_final.m' and uses functions from the other files.
- images : contains the figures and plots of the signal at each stage of the project implementation
- audios : contains the original signal to send and the recovered verion of it
