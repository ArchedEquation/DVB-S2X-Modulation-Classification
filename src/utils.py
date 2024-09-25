import os
import src.DVBS2X as DVBS2X
import random
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.fft import fftshift, fft


def cyclic_autocorrelation(signal, lag):
    return np.mean(signal[:-lag] * np.conjugate(signal[lag:]))


def compute_scd(signal, freqs, alpha):
    """
    Computes the Spectral Correlation Density (SCD) at a given cyclic frequency (alpha).
    """
    N = len(signal)
    scd = np.zeros(len(freqs), dtype=complex)

    for i, f in enumerate(freqs):
        # Shift the signal in frequency by `f`
        shifted_signal = signal * np.exp(-2j * np.pi * f * np.arange(N))

        # Compute cyclic autocorrelation
        cyclic_autocorr = np.mean(
            shifted_signal[:-1] * np.conjugate(shifted_signal[1:]))

        # Apply the exponential factor for the given cyclic frequency alpha
        scd[i] = cyclic_autocorr * np.exp(2j * np.pi * alpha)
    return scd


def save_csv():
    modulation_types = ['bpsk', 'qpsk', '8apsk', '16apsk',
                        '32apsk', '64apsk', '128apsk', '256apsk']
    noise_levels = [5, 10, 15, 20, 25, 30, 35]
    num_symbol_choices = [10, 25, 50, 100]

    features_list = []

    for _ in range(50):  # Generate data for 10 different configurations
        # Randomly select modulation type, noise level, and number of symbols
        modulation = random.choice(modulation_types)
        noise_level = random.choice(noise_levels)
        num_symbols = random.choice(num_symbol_choices)

        dvbs2x = DVBS2X.DVBS2X(num_symbols=num_symbols,
                               samples_per_symbol=8, carrier_freq=1e6)

        modulation_func = {
            'bpsk': dvbs2x.generate_bpsk,
            'qpsk': dvbs2x.generate_qpsk,
            '8apsk': dvbs2x.generate_8apsk,
            '16apsk': dvbs2x.generate_16apsk,
            '32apsk': dvbs2x.generate_32apsk,
            '64apsk': dvbs2x.generate_64apsk,
            '128apsk': dvbs2x.generate_128apsk,
            '256apsk': dvbs2x.generate_256apsk
        }[modulation]

        (t, signal), symbols, bits = modulation_func()

        noisy_signal, noise = dvbs2x.add_noise(signal, noise_level)
        calculated_snr = dvbs2x.calculate_snr(signal, noise)

        lags = range(1, 6)
        alpha = 0.1
        freqs = np.fft.fftfreq(len(signal))

        caf_values = [cyclic_autocorrelation(
            noisy_signal, lag) for lag in lags]
        scd_values = compute_scd(noisy_signal, freqs, alpha)

        for i, symbol in enumerate(symbols):
            magnitude = np.abs(symbol)
            phase = np.angle(symbol)
            # real = np.real(symbol)
            # imag = np.imag(symbol)

            # Create a label based on the bits
            # label = ''.join(map(str, bits[i:i + dvbs2x.num_symbols // len(symbols)]))

            # Create a dictionary to store the features for this symbol
            features = {
                'modulation_type': modulation,
                'symbol': symbol,
                'magnitude': magnitude,
                'phase': phase,
                'caf_1': np.abs(caf_values[0]),  # Example for lag 1
                'caf_2': np.abs(caf_values[1]),  # Example for lag 2
                'caf_3': np.abs(caf_values[2]),  # Example for lag 3
                'caf_4': np.abs(caf_values[3]),  # Example for lag 4
                'caf_5': np.abs(caf_values[4]),  # Example for lag 5
                'scd_mean': np.mean(np.abs(scd_values)),  # Mean of SCD values
                'scd_max': np.max(np.abs(scd_values)),  # Maximum of SCD values
            }

        # Append the features to the list
        features_list.append(features)

    random.shuffle(features_list)
    df = pd.DataFrame(features_list)
    fpath = "symbol_features_1.csv"

    df.to_csv(fpath, index=False)
    csv_path = os.path.abspath(fpath)
    return csv_path
