import numpy as np
import matplotlib.pyplot as plt


class DVBS2X:
    """
    A class to generate DVB-S2X waveforms with various modulation schemes

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate
    samples_per_symbol : int
        Number of samples per symbol
    carrier_freq : float
        Carrier frequency in Hz

    Attributes
    ----------
    num_symbols : int
        Number of symbols to generate
    samples_per_symbol : int
        Number of samples per symbol
    carrier_freq : float
        Carrier frequency in Hz
    """

    def __init__(self, num_symbols, samples_per_symbol, carrier_freq):
        self.num_symbols = num_symbols
        self.samples_per_symbol = samples_per_symbol
        self.carrier_freq = carrier_freq

    def generate_bpsk(self):
        bits = np.random.randint(0, 2, self.num_symbols)
        symbols = 2 * bits - 1  # Map 0 to -1 and 1 to 1
        return self.modulate(symbols), symbols, bits

    def generate_qpsk(self):
        bits = np.random.randint(0, 2, 2 * self.num_symbols)
        qpsk_map = {
            (0, 0): 1 + 1j, (0, 1): -1 + 1j,
            (1, 1): -1 - 1j, (1, 0): 1 - 1j
        }
        symbols = np.array([qpsk_map[tuple(bits[i:i+2])]
                           for i in range(0, len(bits), 2)])
        return self.modulate(symbols), symbols, bits

    def generate_8apsk(self):
        return self._generate_apsk(8, 3, [1, 2.6], [1, 7])

    def generate_16apsk(self):
        return self._generate_apsk(16, 4, [1, 2.6], [4, 12])

    def generate_32apsk(self):
        return self._generate_apsk(32, 5, [1, 2.6, 4.15], [4, 12, 16])

    def generate_64apsk(self):
        return self._generate_apsk(64, 6, [1, 1.6, 2.4, 3.5], [4, 12, 20, 28])

    def generate_128apsk(self):
        return self._generate_apsk(128, 7, [1, 1.5, 2.2, 3.0, 3.8], [4, 12, 20, 40, 52])

    def generate_256apsk(self):
        return self._generate_apsk(256, 8, [1, 1.4, 1.9, 2.5, 3.2, 4.0], [4, 12, 20, 28, 60, 132])

    def _generate_apsk(self, m, bits_per_symbol, radii, points_per_ring):
        bits = np.random.randint(0, 2, bits_per_symbol * self.num_symbols)

        constellation = []
        for r, n in zip(radii, points_per_ring):
            for k in range(n):
                angle = 2 * np.pi * k / n
                constellation.append(r * np.exp(1j * angle))
        constellation = np.array(constellation)

        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))

        symbols = np.zeros(self.num_symbols, dtype=complex)
        for i in range(self.num_symbols):
            bit_chunk = bits[i*bits_per_symbol:(i+1)*bits_per_symbol]
            symbol_index = int(''.join(map(str, bit_chunk)), 2)
            symbols[i] = constellation[symbol_index]

        return self.modulate(symbols), symbols, bits

    def modulate(self, symbols):
        """
        Modulates the given symbols using the carrier frequency and samples_per_symbol.

        Parameters
        ----------
        symbols : array_like
            The complex symbols to be modulated.

        Returns
        -------
        t : array_like
            The time array for the modulated signal.
        signal : array_like
            The modulated signal.
        """
        t = np.arange(self.num_symbols * self.samples_per_symbol) / \
            (self.carrier_freq * self.samples_per_symbol)
        upsampled = np.repeat(symbols, self.samples_per_symbol)
        carrier = np.exp(2j * np.pi * self.carrier_freq * t)
        signal = np.real(upsampled * carrier)

        return t, signal

    def plot_signal(self, t, signal, modulation_type, num_symbols_to_plot=10):
        """
        Plot the given signal for a limited number of symbols.

        Parameters
        ----------
        t : array_like
            The time array of the signal.
        signal : array_like
            The modulated signal.
        modulation_type : str
            The type of modulation used for the signal.
        num_symbols_to_plot : int, optional
            The number of symbols to plot. Defaults to 10.
        """
        samples_to_plot = num_symbols_to_plot * self.samples_per_symbol
        plt.figure(figsize=(12, 6))
        plt.plot(t[:samples_to_plot], signal[:samples_to_plot])
        plt.title(
            f'{modulation_type} Modulated Signal (Zoomed-in on {num_symbols_to_plot} symbols)')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def add_noise(self, signal, snr_db):
        """
        Adds White Gaussian Noise to the given signal based on the desired SNR in dB.

        Parameters
        ----------
        signal : array_like
            The modulated signal.
        snr_db : float
            The desired Signal-to-Noise Ratio (SNR) in decibels.

        Returns
        -------
        noisy_signal : array_like
            The signal with added noise.
        noise : array_like
            The added noise.
        """
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
        noisy_signal = signal + noise
        return noisy_signal, noise

    def calculate_snr(self, signal, noise):
        """
        Calculates the Signal-to-Noise Ratio (SNR) based on the given signal and noise.

        Parameters
        ----------
        signal : array_like
            The modulated signal.
        noise : array_like
            The noise added to the signal.

        Returns
        -------
        snr_db : float
            The calculated SNR in decibels.
        """
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(np.abs(noise)**2)
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db

    def plot_constellation(self, symbols, modulation_type):
        """
        Plots the constellation diagram of the given symbols and modulation type.

        Parameters
        ----------
        symbols : array_like
            The modulated symbols.
        modulation_type : str
            The type of modulation used for the symbols.

        Returns
        -------
        None
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(symbols.real, symbols.imag, c='r', alpha=0.5)
        plt.title(f'{modulation_type} Constellation Diagram')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


def demo():
    dvbs2x = DVBS2X(num_symbols=1000, samples_per_symbol=8, carrier_freq=1e6)

    modulation_schemes = [
        ('BPSK', dvbs2x.generate_bpsk),
        ('QPSK', dvbs2x.generate_qpsk),
        ('8-APSK', dvbs2x.generate_8apsk),
        ('16-APSK', dvbs2x.generate_16apsk),
        ('32-APSK', dvbs2x.generate_32apsk),
        ('64-APSK', dvbs2x.generate_64apsk),
        ('128-APSK', dvbs2x.generate_128apsk),
        ('256-APSK', dvbs2x.generate_256apsk)
    ]

    snr_db = 10
    for name, func in modulation_schemes:
        (t, signal), symbols, bits = func()
        noisy_signal, noise = dvbs2x.add_noise(signal, snr_db)

        calculated_snr = dvbs2x.calculate_snr(signal, noise)
        print(f"\n{name} - Calculated SNR: {calculated_snr:.2f} dB")
        print(f"{name} - First 10 symbols: {symbols[:10]}")

        dvbs2x.plot_signal(t, noisy_signal, name)
        # dvbs2x.plot_constellation(symbols, name)
