"""
Generate CW (Continuous Wave) I/Q samples for testing CoolSDR
This script creates a CSV file with I/Q samples of a CW signal
"""

import numpy as np
import csv

def generate_cw_iq_samples(duration=3, sample_rate=48000, carrier_freq=1000, cw_pattern=None):
    """
    Generate CW I/Q samples
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        carrier_freq: Carrier frequency in Hz
        cw_pattern: Pattern of on/off times in seconds (list of tuples: (on_time, off_time))
                   If None, uses default morse-like pattern
    
    Returns:
        Tuple of (i_samples, q_samples)
    """
    
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # Create carrier
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    
    # Create CW envelope (on/off pattern)
    if cw_pattern is None:
        # Default: morse-like pattern (dit=0.1s, dah=0.3s, space=0.1s, letter_space=0.3s, word_space=0.7s)
        # Pattern: "CW" in morse code
        # C = dah-dit-dah-dit
        # W = dah-dah-dit
        cw_pattern = [
            (0.3, 0.1),  # C: dah
            (0.1, 0.1),  # dit
            (0.3, 0.1),  # dah
            (0.1, 0.3),  # dit + letter space
            (0.3, 0.1),  # W: dah
            (0.3, 0.1),  # dah
            (0.1, 0.5),  # dit + word space
        ]
    
    envelope = np.zeros(n_samples)
    current_time = 0.0
    
    for on_time, off_time in cw_pattern:
        on_samples = int(on_time * sample_rate)
        off_samples = int(off_time * sample_rate)
        
        # Fill in the "on" portion
        start_idx = int(current_time * sample_rate)
        end_idx = min(start_idx + on_samples, n_samples)
        envelope[start_idx:end_idx] = 1.0
        
        current_time += on_time + off_time
        
        if current_time * sample_rate >= n_samples:
            break
    
    # Apply envelope to carrier
    i_samples = carrier_i * envelope
    q_samples = carrier_q * envelope
    
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
    # Generate CW samples
    print("Generating CW I/Q samples...")
    i_samples, q_samples = generate_cw_iq_samples(
        duration=3,           # 3 seconds
        sample_rate=48000,    # 48 kHz (matches CoolSDR)
        carrier_freq=1000,    # 1 kHz carrier
    )
    
    # Save to CSV
    csv_filename = "cw_test_samples.csv"
    save_iq_csv(csv_filename, i_samples, q_samples)
    
    print(f"\nGenerated CW test file: {csv_filename}")
    print(f"Total samples: {len(i_samples)}")
    print(f"Duration: 3 seconds")
    print(f"Sample rate: 48000 Hz")
    print(f"Carrier frequency: 1000 Hz")
    print("\nUsage in CoolSDR:")
    print("1. Select 'CSV File (Debug)' from I/Q Source dropdown")
    print("2. Click 'Load CSV File' and select cw_test_samples.csv")
    print("3. Select 'CW' modulation type")
    print("4. Click 'Start' to test demodulation")
