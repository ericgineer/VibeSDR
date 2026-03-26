"""
Generate SSB (Single Sideband) I/Q samples for testing CoolSDR
This script creates a CSV file with I/Q samples of an SSB signal
"""

import numpy as np
import csv
from scipy import signal

def generate_audio_signal(duration=3, sample_rate=48000, frequencies=None):
    """
    Generate audio test signal (combination of sinusoids)
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequencies: List of frequencies in Hz. If None, uses [1000, 2000]
    
    Returns:
        Audio samples
    """
    if frequencies is None:
        frequencies = [1000, 2000]
    
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    audio = np.zeros(n_samples)
    for freq in frequencies:
        audio += np.sin(2 * np.pi * freq * t)
    
    # Normalize
    audio = audio / (len(frequencies) * 1.2)
    
    return audio


def generate_ssb_iq_samples(duration=3, sample_rate=48000, carrier_freq=0, 
                             audio_freqs=None, sideband='USB'):
    """
    Generate SSB (Single Sideband) I/Q samples
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        carrier_freq: Carrier frequency in Hz (typically 0 for baseband)
        audio_freqs: List of audio frequencies to modulate
        sideband: 'USB' (Upper SideBand) or 'LSB' (Lower SideBand)
    
    Returns:
        Tuple of (i_samples, q_samples)
    """
    
    # Generate audio modulating signal
    audio = generate_audio_signal(duration, sample_rate, audio_freqs)
    
    n_samples = len(audio)
    
    # For SSB, compute the Hilbert transform of the audio signal
    # This creates the analytic signal (complex representation)
    audio_analytic = signal.hilbert(audio)
    
    # Create the SSB signal by mixing with carrier
    t = np.arange(n_samples) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    if sideband.upper() == 'USB':
        # Upper SideBand: multiply by carrier
        ssb_complex = audio_analytic * carrier
    else:
        # Lower SideBand: multiply by conjugate of carrier
        ssb_complex = np.conj(audio_analytic) * carrier
    
    # Extract I and Q components
    i_samples = np.real(ssb_complex)
    q_samples = np.imag(ssb_complex)
    
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
    # Generate SSB samples with two audio tones
    print("Generating SSB I/Q samples (USB)...")
    i_samples, q_samples = generate_ssb_iq_samples(
        duration=3,                    # 3 seconds
        sample_rate=48000,             # 48 kHz (matches CoolSDR)
        carrier_freq=0,                # Baseband (0 Hz carrier)
        audio_freqs=[1000, 2000],      # 1 kHz and 2 kHz tones
        sideband='USB'                 # Upper SideBand
    )
    
    # Save to CSV
    csv_filename = "ssb_test_samples.csv"
    save_iq_csv(csv_filename, i_samples, q_samples)
    
    print(f"\nGenerated SSB test file: {csv_filename}")
    print(f"Total samples: {len(i_samples)}")
    print(f"Duration: 3 seconds")
    print(f"Sample rate: 48000 Hz")
    print(f"Carrier frequency: 0 Hz (baseband)")
    print(f"Audio frequencies: 1 kHz + 2 kHz")
    print(f"Sideband: USB (Upper SideBand)")
    print("\nUsage in CoolSDR:")
    print("1. Select 'CSV File (Debug)' from I/Q Source dropdown")
    print("2. Click 'Load CSV File' and select ssb_test_samples.csv")
    print("3. Select 'SSB' modulation type")
    print("4. Click 'Start' to test demodulation")
    print("\nYou should hear two tones at 1 kHz and 2 kHz")
