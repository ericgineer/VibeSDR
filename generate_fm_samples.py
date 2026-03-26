"""
Generate FM (Frequency Modulation) I/Q samples for testing CoolSDR
This script creates a CSV file with I/Q samples of an FM signal
"""

import numpy as np
import csv

def generate_audio_signal(duration=3, sample_rate=48000, frequencies=None):
    """
    Generate audio test signal (combination of sinusoids)
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequencies: List of frequencies in Hz. If None, uses [1000]
    
    Returns:
        Audio samples
    """
    if frequencies is None:
        frequencies = [1000]
    
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    audio = np.zeros(n_samples)
    for freq in frequencies:
        audio += np.sin(2 * np.pi * freq * t)
    
    # Normalize
    audio = audio / (len(frequencies) * 1.2)
    
    return audio


def generate_fm_iq_samples(duration=3, sample_rate=48000, carrier_freq=1000, 
                            audio_freqs=None, frequency_deviation=2000):
    """
    Generate FM (Frequency Modulation) I/Q samples
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        carrier_freq: Carrier frequency in Hz
        audio_freqs: List of audio frequencies to modulate
        frequency_deviation: Maximum frequency deviation in Hz
                            (Total FM bandwidth = 2 * (frequency_deviation + audio_freq))
                            For audio_freq=1000 Hz and dev=2000 Hz, BW = 6000 Hz
    
    Returns:
        Tuple of (i_samples, q_samples)
    """
    
    # Generate audio modulating signal
    audio = generate_audio_signal(duration, sample_rate, audio_freqs)
    
    n_samples = len(audio)
    t = np.arange(n_samples) / sample_rate
    dt = 1.0 / sample_rate
    
    # FM: s(t) = cos(2*pi*f_c*t + 2*pi*Delta_f*integral(m(t)dt))
    # Compute the integral of the modulating signal
    modulation_integral = np.cumsum(audio) * dt
    
    # Compute the instantaneous phase
    # phase(t) = 2*pi*f_c*t + 2*pi*Delta_f*integral(m(t)dt)
    phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * frequency_deviation * modulation_integral
    
    # Generate I/Q components
    i_samples = np.cos(phase)
    q_samples = np.sin(phase)
    
    return i_samples, q_samples


def save_iq_csv(filename, i_samples, q_samples):
    """Save I/Q samples to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['I', 'Q'])  # Header
        for i, q in zip(i_samples, q_samples):
            writer.writerow([i, q])
    print(f"Saved {len(i_samples)} I/Q samples to {filename}")


if __name__ == "__main__":
    # Generate FM samples with one audio tone
    print("Generating FM I/Q samples...")
    i_samples, q_samples = generate_fm_iq_samples(
        duration=3,                    # 3 seconds
        sample_rate=48000,             # 48 kHz (matches CoolSDR)
        carrier_freq=1000,             # 1 kHz carrier
        audio_freqs=[1000],            # 1 kHz audio tone
        frequency_deviation=2000       # 2 kHz frequency deviation
    )
    
    # Save to CSV
    csv_filename = "fm_test_samples.csv"
    save_iq_csv(csv_filename, i_samples, q_samples)
    
    print(f"\nGenerated FM test file: {csv_filename}")
    print(f"Total samples: {len(i_samples)}")
    print(f"Duration: 3 seconds")
    print(f"Sample rate: 48000 Hz")
    print(f"Carrier frequency: 1000 Hz")
    print(f"Audio frequency: 1000 Hz")
    print(f"Frequency deviation: 2000 Hz")
    print(f"FM bandwidth (Carson's Rule): ~6000 Hz")
    print("\nUsage in CoolSDR:")
    print("1. Select 'CSV File (Debug)' from I/Q Source dropdown")
    print("2. Click 'Load CSV File' and select fm_test_samples.csv")
    print("3. Select 'FM' modulation type")
    print("4. Click 'Start' to test demodulation")
    print("\nYou should hear a 1 kHz tone as the demodulated audio")
