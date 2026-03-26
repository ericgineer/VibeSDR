"""
Generate AM (Amplitude Modulation) I/Q samples for testing CoolSDR
This script creates a CSV file with I/Q samples of an AM signal
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


def generate_am_iq_samples(duration=3, sample_rate=48000, carrier_freq=1000, 
                            audio_freqs=None, modulation_index=0.8):
    """
    Generate AM (Amplitude Modulation) I/Q samples
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        carrier_freq: Carrier frequency in Hz
        audio_freqs: List of audio frequencies to modulate
        modulation_index: Modulation depth (0.0 to 1.0)
                         0.8 is typical (80% modulation)
    
    Returns:
        Tuple of (i_samples, q_samples)
    """
    
    # Generate audio modulating signal
    audio = generate_audio_signal(duration, sample_rate, audio_freqs)
    
    n_samples = len(audio)
    t = np.arange(n_samples) / sample_rate
    
    # AM: s(t) = [1 + m(t)] * cos(2*pi*f_c*t)
    # where m(t) is the modulating signal (audio)
    # The modulation index controls the depth of modulation
    modulated = 1.0 + modulation_index * audio
    
    # Create carrier
    carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
    carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
    
    # Modulate carrier by multiplying with envelope
    i_samples = modulated * carrier_cos
    q_samples = modulated * carrier_sin
    
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
    # Generate AM samples with one audio tone
    print("Generating AM I/Q samples...")
    i_samples, q_samples = generate_am_iq_samples(
        duration=3,                    # 3 seconds
        sample_rate=48000,             # 48 kHz (matches CoolSDR)
        carrier_freq=1000,             # 1 kHz carrier
        audio_freqs=[500],             # 500 Hz audio tone
        modulation_index=0.8           # 80% modulation
    )
    
    # Save to CSV
    csv_filename = "am_test_samples.csv"
    save_iq_csv(csv_filename, i_samples, q_samples)
    
    print(f"\nGenerated AM test file: {csv_filename}")
    print(f"Total samples: {len(i_samples)}")
    print(f"Duration: 3 seconds")
    print(f"Sample rate: 48000 Hz")
    print(f"Carrier frequency: 1000 Hz")
    print(f"Audio frequency: 500 Hz")
    print(f"Modulation index: 80%")
    print("\nUsage in CoolSDR:")
    print("1. Select 'CSV File (Debug)' from I/Q Source dropdown")
    print("2. Click 'Load CSV File' and select am_test_samples.csv")
    print("3. Select 'AM' modulation type")
    print("4. Click 'Start' to test demodulation")
    print("\nYou should hear a 500 Hz tone as the demodulated audio")
