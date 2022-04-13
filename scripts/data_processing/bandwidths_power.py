import numpy as np 


def bandwidths_power(time_signal, sampling_frequency, time_period, time_window, time_shift, bandwidths):

    # for each sample (i.e time step), substract the mean across all units
    time_signal = time_signal - time_signal.mean(axis=1)[:, np.newaxis] 

    # truncate signals given the number of samples to consider
    if time_period > 0:
        n_samples = time_period * sampling_frequency 
    else: 
        n_samples = time_signal.shape[0]

    time_signal = time_signal[-n_samples:, :] 

    sample_window = int(time_window * sampling_frequency)
    sample_shift = int(time_shift * sampling_frequency) 

    # frequencies at which the FFT is computed
    frequency_space = np.fft.rfftfreq(sample_window, 1 / sampling_frequency) 

    # number of frames for which the FFT is computed
    n_frames = int((n_samples - sample_window) / sample_shift) + 1 
    n_units = time_signal.shape[1]
    n_bandwidths = len(bandwidths)

    bandwidths_power = np.zeros((n_frames, n_units, n_bandwidths)) 

    # for each frame, compute the power spectrum on the given window
    for f in range(n_frames): 

        sample_start = f * sample_shift

        signal_window = time_signal[sample_start: sample_start + sample_window, :]
        signal_window = signal_window - signal_window.mean(0) # substract mean to avoid zero-drift

        power_spectrum = _compute_power_spectrum(signal_window, frequency_space[1])

        # for each bandwith, sum power spectrum over desired frequencies
        for b, bandwidth in enumerate(bandwidths): 

            lower_bound = bandwidth[0]
            upper_bound = bandwidth[1]

            bandwidth_mask = (frequency_space >= lower_bound) & (frequency_space < upper_bound)
            bandwidth_power = power_spectrum[bandwidth_mask, :].sum(0) 

            bandwidths_power[f, :, b] = bandwidth_power

    return bandwidths_power

def _compute_power_spectrum(signal, bin_width):

    freq_amplitude = np.abs(np.fft.fft(signal, axis=0))
    freq_amplitude = freq_amplitude / freq_amplitude.shape[0] # divide by number of entries

    power_spectrum = freq_amplitude ** 2
    power_spectrum = power_spectrum / bin_width # divide by bin width to get density

    if power_spectrum.shape[0] % 2 == 0: # odd shape 
            cut = int(2 + (power_spectrum.shape[0]-1) / 2)

    else: # even shape 
        cut = int(2 + power_spectrum.shape[0] / 2)

    # consider only one-side of power spectrum, signal is real
    power_spectrum = power_spectrum[:cut] 

    return power_spectrum