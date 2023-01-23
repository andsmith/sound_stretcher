import numpy as np
import logging
from sound import Sound
from spectrograms import get_power_spectrum
import matplotlib.pyplot as plt

def test_spectrogram():
    """
    Generate spectrum of random data, make sure different chunk sizes for the piece-wise SFTF are identical.
    """
    data = np.random.randn(100000)
    max_fft_sizes = [10000, 20000, 60000,100000,200000]

    results = [get_power_spectrum(data, 44100, resolution_hz=100., MAX_FFT_SIZE=mfs)[0] for mfs in max_fft_sizes]
    for i, result in enumerate(results[:-1]):
        logging.info("Comparing result:  MAX_FFT_SIZE: %i to %i" %(max_fft_sizes[i], max_fft_sizes[i+1]))
        assert (np.max(np.abs(result - results[i + 1])) < 1e-10), "Results with MAX_FFT_SIZE = %i and %i differ."  % (max_fft_sizes[i], max_fft_sizes[i+1])

def _debug_sandbox():

    logging.basicConfig(level=logging.INFO)
    sound = Sound("Aphelocoma_californica_-_California_Scrub_Jay_-_XC110976.wav")

    z, f, t = get_power_spectrum(sound.get_mono_data()[:100000], sound.metadata.framerate, MAX_FFT_SIZE=20000)
    z2, f2, t2 = get_power_spectrum(sound.get_mono_data()[:100000], sound.metadata.framerate, MAX_FFT_SIZE=30000)

    plt.plot(np.diff(t))
    plt.show()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(np.log(4 + np.abs(z.T)), aspect='auto', origin='lower')
    plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
    plt.imshow(np.log(4 + np.abs(z2.T)), aspect='auto', origin='lower')
    plt.show()
    plt.imshow(np.log(4 + np.abs(z2.T))-np.log(4 + np.abs(z.T)), aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_spectrogram()
    print("All tests pass.")
