import time
import random
import pandas as pd
import numpy as np

from src.DVBS2X import DVBS2X
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from src.utils import cyclic_autocorrelation, compute_scd, weighted_score


start_time = time.time()
if __name__ == "__main__":
    # Loop to randomly generate data
    for _ in range(50):  # Generate data for 10 different configurations
        # Randomly select modulation type, noise level, and number of symbols
        modulation_types = ['bpsk', 'qpsk', '8apsk', '16apsk',
                            '32apsk', '64apsk', '128apsk', '256apsk']
        noise_levels = [5, 10, 15, 20, 25, 30, 35]
        num_symbol_choices = [10, 25, 50, 100]

        # Create a list to store the features for each symbol
        features_list = []
        modulation = random.choice(modulation_types)
        noise_level = random.choice(noise_levels)
        num_symbols = random.choice(num_symbol_choices)

        # Initialize the DVBS2X class with the selected number of symbols
        dvbs2x = DVBS2X(num_symbols=num_symbols,
                        samples_per_symbol=8, carrier_freq=1e6)

        # Select the appropriate modulation function based on the modulation type
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
            real = np.real(symbol)
            imag = np.imag(symbol)

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
            }

            # Append the features to the list
            features_list.append(features)
    random.shuffle(features_list)

    # Create a pandas DataFrame from the list of features
    df = pd.DataFrame(features_list)
    # print(df.head())

    # Save the DataFrame to a CSV file
    df.to_csv('data/train.csv', index=False)

    data = pd.read_csv("data/train.csv")
    label_enc = LabelEncoder()
    df['modulation'] = label_enc.fit_transform(df['modulation_type'])

    X = df[['magnitude', 'phase', 'caf_1', 'caf_2',
            'caf_3', 'caf_4', 'caf_5', 'scd_mean']]
    y = df['modulation']

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    for i in range(25):
        modulation_types = ['bpsk', '8apsk', '16apsk',
                            '32apsk', '256apsk']
        noise_levels = [5, 10, 25, 35]
        num_symbol_choices = [10, 50, 100]
        features_list = []
        modulation = random.choice(modulation_types)
        noise_level = random.choice(noise_levels)
        num_symbols = random.choice(num_symbol_choices)

        # Initialize the DVBS2X class with the selected number of symbols
        dvbs2x = DVBS2X(num_symbols=num_symbols,
                        samples_per_symbol=8, carrier_freq=1e6)

        # Select the appropriate modulation function based on the modulation type
        test_modulation = {
            'bpsk': dvbs2x.generate_bpsk,
            '8apsk': dvbs2x.generate_8apsk,
            '16apsk': dvbs2x.generate_16apsk,
            '32apsk': dvbs2x.generate_32apsk,
            '256apsk': dvbs2x.generate_256apsk
        }[modulation]

        (t, signal), symbols, bits = test_modulation()

        noisy_signal, noise = dvbs2x.add_noise(signal, noise_level)
        calculated_snr = dvbs2x.calculate_snr(signal, noise)

        lags = range(1, 6)
        alpha = 0.25
        freqs = np.fft.fftfreq(len(signal))

        caf_values = [cyclic_autocorrelation(
            noisy_signal, lag) for lag in lags]
        scd_values = compute_scd(noisy_signal, freqs, alpha)

        for i, symbol in enumerate(symbols):
            magnitude = np.abs(symbol)
            phase = np.angle(symbol)

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
            }

            # Append the features to the list
            features_list.append(features)
    random.shuffle(features_list)

    # Create a pandas DataFrame from the list of features
    df = pd.DataFrame(features_list)
    # print(df.head())

    # Save the DataFrame to a CSV file
    df.to_csv('data/test.csv', index=False)

    df = pd.read_csv("data/test.csv")
    label_enc = LabelEncoder()
    df['modulation'] = label_enc.fit_transform(df['modulation_type'])

    X_t = df[['magnitude', 'phase', 'caf_1', 'caf_2',
              'caf_3', 'caf_4', 'caf_5', 'scd_mean']]
    y_t = df['modulation']

    y_pred = rf_model.predict(X_t)
    rf_model.score(X_t, y_t)

    rf_accuracy = accuracy_score(y_t, y_pred)
    print("\nRandom Forest Model Accuracy: {:.2f}%".format(rf_accuracy * 100))
    print("\nClassification Report for Random Forest:")
    print(classification_report(y_t, y_pred))
    end_time = time.time()

print(weighted_score(model=rf_model, start_time=start_time,
      end_time=end_time, X_test=X_t, y_test=y_t))
