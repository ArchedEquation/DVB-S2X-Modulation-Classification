import pickle
import numpy as np


def cyclic_autocorrelation(signal, lag):
    """
    Computes the cyclic autocorrelation of a signal at a given lag.

    Parameters
    ----------
    signal : array_like
        The input signal.
    lag : int
        The lag at which to compute the cyclic autocorrelation.

    Returns
    -------
    acf : complex
        The cyclic autocorrelation of the signal at the given lag.
    """
    return np.mean(signal[:-lag] * np.conjugate(signal[lag:]))


def compute_scd(signal, freqs, alpha):
    """
    Computes the spectral correlation density (SCD) of a signal at a given cyclic frequency `alpha`.

    Parameters
    ----------
    signal : array_like
        The input signal.
    freqs : array_like
        The frequencies at which to compute the SCD.
    alpha : float
        The cyclic frequency at which to compute the SCD.

    Returns
    -------
    scd : array_like
        The SCD of the signal at the given frequencies and cyclic frequency.
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


def weighted_score(model, start_time, end_time, X_test, y_test, weight_for_latency=0.2, weight_for_model_size=0.1):
    """
    Computes a weighted score for a given model, taking into account its latency,
    accuracy, and model size.

    Parameters
    ----------
    model : object
        The model object to be scored.
    start_time : float
        The start time of the model inference.
    end_time : float
        The end time of the model inference.
    X_test : array_like
        The input data for the model.
    y_test : array_like
        The true labels for the model.
    weight_for_latency : float, optional
        The weight for the latency score. Default is 0.2.
    weight_for_model_size : float, optional
        The weight for the model size score. Default is 0.1.

    Returns
    -------
    score : float
        The weighted score for the model.
    """
    try:
        accuracy = model.score(X_test, y_test)
        _ = model.predict(X_test)
        latency = (end_time - start_time) / len(X_test)

        model_pickle = pickle.dumps(model)
        model_size = len(model_pickle)

        max_latency = 0.01
        max_model_size = 1e6

        normalized_latency = latency/max_latency
        normalized_size = model_size/max_model_size

        score = (accuracy) - (weight_for_latency *
                              normalized_latency) - (weight_for_model_size * normalized_size)

        result_string = (
            f"Accuracy: {accuracy:.3f}\n"
            f"Normalized Latency: {
                normalized_latency:.3f} and Latency: {latency}\n"
            f"Normalized Size: {normalized_size:.3f} and Size: {
                model_size} in Bytes\n"
            f"Final Power-Efficient Score: {score:.3f}"
        )

        return result_string

    except Exception as e:
        print(f"Error calculating power-efficient score: {e}")
        return None
