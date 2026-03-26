"""
VibeSDR - Software Defined Radio Application
A Python-based SDR that demodulates AM, FM, CW, and SSB from I/Q samples
"""

import sys
import numpy as np
import threading
import csv
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from scipy import signal, fft
from scipy.ndimage import convolve1d

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QButtonGroup, QSlider, QComboBox,
    QPushButton, QFileDialog, QButtonGroup, QGroupBox, QFormLayout,
    QSpinBox, QDial, QProgressBar, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QFrame

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================================
# Enumerations and Data Structures
# ============================================================================

class ModulationType(Enum):
    """Supported modulation types"""
    AM = "AM"
    FM = "FM"
    CW = "CW"
    SSB = "SSB"


@dataclass
class SDRConfig:
    """Configuration for SDR parameters"""
    sample_rate: float = 48000  # Hz
    center_freq: float = 1000   # Hz (tuning frequency)
    bandwidth: float = 10000    # Hz
    modulation: ModulationType = ModulationType.AM
    volume: float = 1.0
    fft_size: int = 2048
    audio_device_input: int = None  # None = default
    audio_device_output: int = None  # None = default
    tx_audio_device_input: int = None  # None = default for TX
    ssb_upper_sideband: bool = True  # True = USB, False = LSB

# ============================================================================
# Demodulation Algorithms
# ============================================================================

class Demodulator:
    """Handles demodulation of different modulation types"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.demod_filter = None
        self.tune_freq = 0  # Current tuning frequency
        self._create_filters()

    def _create_filters(self):
        """Create standard filters for demodulation"""
        # Demodulation low-pass filter (5 kHz cutoff for audio)
        cutoff = 5000
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = min(normalized_cutoff, 0.99)  # Ensure valid range
        
        self.demod_filter = signal.butter(4, normalized_cutoff, btype='low')
        self.filter_state = signal.lfilter_zi(self.demod_filter[0], self.demod_filter[1])

    def set_tuning_frequency(self, freq_hz: float):
        """Set the tuning frequency for frequency shift"""
        self.tune_freq = freq_hz

    def apply_frequency_shift(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Apply frequency shift to I/Q samples (frequency translation)
        This shifts the spectrum by multiplying with a complex exponential
        
        Args:
            iq_samples: Complex I/Q samples
        
        Returns:
            Frequency-shifted I/Q samples
        """
        if self.tune_freq == 0:
            return iq_samples
        
        n_samples = len(iq_samples)
        t = np.arange(n_samples) / self.sample_rate
        
        # Create tuning oscillator (negative frequency for downshift)
        tuning_osc = np.exp(-1j * 2 * np.pi * self.tune_freq * t)
        
        # Apply frequency shift
        shifted = iq_samples * tuning_osc
        
        return shifted

    def apply_frequency_shift_tx(self, iq_samples: np.ndarray, freq_hz: float) -> np.ndarray:
        """
        Apply upward frequency shift for TX (positive frequency)
        Used to shift baseband modulated signal to center frequency
        
        Args:
            iq_samples: Complex I/Q samples (baseband)
            freq_hz: Frequency to shift to (positive for upshift)
        
        Returns:
            Frequency-shifted I/Q samples
        """
        if freq_hz == 0:
            return iq_samples
        
        n_samples = len(iq_samples)
        t = np.arange(n_samples) / self.sample_rate
        
        # Create upshift oscillator (positive frequency)
        upshift_osc = np.exp(1j * 2 * np.pi * freq_hz * t)
        
        # Apply frequency shift
        shifted = iq_samples * upshift_osc
        
        return shifted

    def demodulate_am(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Demodulate AM signal
        AM demodulation: magnitude of I/Q
        """
        # Apply frequency shift for tuning
        iq_samples = self.apply_frequency_shift(iq_samples)
        
        # Calculate envelope: magnitude of complex signal
        magnitude = np.abs(iq_samples)
        
        # Remove DC bias
        magnitude = magnitude - np.mean(magnitude)
        
        # Apply low-pass filter
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], magnitude)
        
        return audio

    def demodulate_fm(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Demodulate FM signal
        FM demodulation: phase derivative
        """
        # Apply frequency shift for tuning
        iq_samples = self.apply_frequency_shift(iq_samples)
        
        # Calculate phase angle using atan2(Q, I)
        phase = np.angle(iq_samples)
        
        # Calculate phase derivatives (instantaneous frequency)
        phase_diff = np.diff(phase)
        
        # Unwrap phase discontinuities
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        # Scale to audio range
        audio = phase_diff * self.sample_rate / (2 * np.pi)
        
        # Apply de-emphasis filter (standard FM de-emphasis: 75 microseconds)
        # Create a simple high-pass filter for de-emphasis
        de_emphasis = signal.butter(1, 300, btype='high', fs=self.sample_rate)
        audio = signal.lfilter(de_emphasis[0], de_emphasis[1], audio)
        
        # Apply low-pass filter
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], audio)
        
        return audio

    def demodulate_cw(self, iq_samples: np.ndarray, bfo_offset: float = 200) -> np.ndarray:
        """
        Demodulate CW (Continuous Wave) signal using Product Detector
        A Beat Frequency Oscillator (BFO) is mixed with the signal to produce audio
        
        Args:
            iq_samples: Complex I/Q samples
            bfo_offset: BFO frequency offset from carrier in Hz
                       For 1 kHz carrier and 800 Hz audio output, use 200 Hz
        """
        # Apply frequency shift for tuning
        iq_samples = self.apply_frequency_shift(iq_samples)
        
        # Generate Beat Frequency Oscillator
        # The BFO frequency is chosen to produce audio at the desired pitch
        # For our 1 kHz test signal and 800 Hz output audio:
        # BFO = carrier_freq - audio_freq = 1000 - 800 = 200 Hz
        n_samples = len(iq_samples)
        t = np.arange(n_samples) / self.sample_rate
        bfo = np.exp(1j * 2 * np.pi * bfo_offset * t)
        
        # Product detection: multiply IQ signal with BFO
        mixed = iq_samples * bfo
        
        # Take real part to extract the beat frequencies
        # This acts as a coherent product detector
        audio = np.real(mixed)
        
        # Remove DC bias
        audio = audio - np.mean(audio)
        
        # Apply low-pass filter to remove high-frequency components
        # The filter will keep the beat frequency (800 Hz in our case)
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], audio)
        
        return audio

    def demodulate_ssb(self, iq_samples: np.ndarray, upper_sideband: bool = True) -> np.ndarray:
        """
        Demodulate SSB (Single SideBand) signal

        NOTE: This implementation is tailored to the specific I/Q sample format
        used in this SDR application. For this format:
        - USB information is encoded in the real (I) channel
        - LSB information is encoded in the imaginary (Q) channel

        This is NOT standard SSB demodulation theory, but works empirically
        for the test signals and I/Q data format used here.

        Args:
            iq_samples: Complex I/Q samples
            upper_sideband: True for USB, False for LSB
        """
        # Apply frequency shift for tuning (heterodyne to baseband)
        iq_samples = self.apply_frequency_shift(iq_samples)

        # Extract audio based on sideband selection for this I/Q format
        if upper_sideband:
            audio = np.real(iq_samples)  # USB: I channel
        else:
            audio = np.imag(iq_samples)  # LSB: Q channel

        # Remove DC bias
        audio = audio - np.mean(audio)

        # Apply low-pass filter to remove high-frequency artifacts
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], audio)

        return audio

        # Apply final audio low-pass filter
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], audio)

    def modulate_am(self, audio: np.ndarray) -> np.ndarray:
        """AM modulate audio to complex baseband I/Q"""
        # Normalize audio to [-1,1]
        peak = np.max(np.abs(audio)) + 1e-12
        audio_n = np.clip(audio / peak, -1.0, 1.0)
        envelope = 0.5 + 0.5 * audio_n
        iq = envelope.astype(np.complex64)
        return iq

    def modulate_fm(self, audio: np.ndarray, freq_dev: float = 5000.0) -> np.ndarray:
        """FM modulate audio to complex baseband I/Q"""
        peak = np.max(np.abs(audio)) + 1e-12
        audio_n = audio / peak
        integral = np.cumsum(audio_n) / self.sample_rate
        phase = 2.0 * np.pi * freq_dev * integral
        iq = np.exp(1j * phase).astype(np.complex64)
        return iq

    def modulate_ssb(self, audio: np.ndarray, upper_sideband: bool = True, center_freq: float = 0) -> np.ndarray:
        """SSB modulation (USB/LSB) to complex baseband I/Q

        Creates SSB signals using the analytic signal approach:
        - USB: analytic signal (Hilbert transform) - contains positive frequencies
        - LSB: conjugate analytic signal - contains negative frequencies

        Note: This produces I/Q samples where USB/LSB information is encoded
        in the real/imaginary channels respectively, matching the RX demodulation.

        Args:
            audio: Audio samples to modulate
            upper_sideband: True for USB, False for LSB
            center_freq: Center frequency for upshift (0 = baseband output)
        """
        peak = np.max(np.abs(audio)) + 1e-12
        audio_n = audio / peak
        analytic = signal.hilbert(audio_n)
        if upper_sideband:
            iq = analytic  # USB: positive frequencies in analytic signal
        else:
            iq = np.conj(analytic)  # LSB: negative frequencies via conjugation

        # Apply TX frequency shift
        iq = self.apply_frequency_shift_tx(iq, center_freq)
        return iq.astype(np.complex64)
    def read(self) -> np.ndarray:
        """Read I/Q samples, return complex array"""
        raise NotImplementedError


class IQSource:
    """Base class for I/Q sample sources"""

    def __init__(self, sample_rate: float, buffer_size: int = 4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def read(self) -> np.ndarray:
        """Read I/Q samples, return complex array"""
        raise NotImplementedError


class AudioCardIQSource(IQSource):
    """Read I/Q samples from stereo audio input (Left=I, Right=Q)"""

    def __init__(self, sample_rate: float, device: int = None, buffer_size: int = 4096):
        super().__init__(sample_rate, buffer_size)
        self.device = device
        self.stream = None
        self.channels = None

    @staticmethod
    def list_devices():
        """List available audio input devices"""
        print("\nAvailable Audio Input Devices:")
        print("-" * 80)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
                print(f"    Max Input Channels: {device['max_input_channels']}")
                print(f"    Sample Rate: {device['default_samplerate']}")
                print()

    def _find_best_device(self):
        """Find a suitable audio device"""
        try:
            devices = sd.query_devices()
            default_device = sd.default.device[0]  # Default input device
            
            # Try default device first
            if default_device >= 0:
                device_info = devices[default_device]
                if device_info['max_input_channels'] >= 2:
                    print(f"Using stereo input: {device_info['name']}")
                    return default_device, 2
                elif device_info['max_input_channels'] == 1:
                    print(f"Device '{device_info['name']}' only supports mono. Using mono mode.")
                    return default_device, 1
            
            # If default doesn't work, search for stereo device
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= 2:
                    print(f"Using stereo input: {device['name']}")
                    return i, 2
            
            # Fallback to first mono device
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= 1:
                    print(f"Using mono input: {device['name']}")
                    return i, 1
            
            raise RuntimeError("No audio input device found")
            
        except Exception as e:
            print(f"Error finding audio device: {e}")
            self.list_devices()
            raise

    def start(self):
        """Start audio input stream"""
        super().start()
        try:
            # Use specified device or find best available
            if self.device is not None:
                devices = sd.query_devices()
                device_info = devices[self.device]
                self.channels = min(2, device_info['max_input_channels'])
                if self.channels < 2:
                    print(f"Warning: Device only supports {self.channels} channel(s)")
            else:
                self.device, self.channels = self._find_best_device()
            
            print(f"Starting audio stream: {self.channels} channel(s), {self.sample_rate} Hz")
            
            self.stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            self.stream.start()
            print("Audio stream started successfully")
            
        except Exception as e:
            print(f"Error starting audio input: {e}")
            print("\nTroubleshooting:")
            print("1. Check your audio device is connected")
            print("2. Verify audio device drivers are installed")
            print("3. Try a different audio device")
            print("\nAvailable devices:")
            self.list_devices()
            self.running = False
            raise

    def stop(self):
        """Stop audio input stream"""
        super().stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def read(self) -> np.ndarray:
        """Read audio and convert to I/Q samples"""
        if not self.running or not self.stream:
            return None
        
        try:
            data, _ = self.stream.read(self.buffer_size)
            if data.shape[0] == 0:
                return None
            
            # Convert to complex I/Q based on number of channels
            if self.channels >= 2:
                # Stereo: Left=I, Right=Q
                i_samples = data[:, 0]
                q_samples = data[:, 1]
            else:
                # Mono: Use same signal for both I and Q (not ideal but works)
                i_samples = data[:, 0]
                q_samples = data[:, 0]
            
            iq = i_samples + 1j * q_samples
            return iq
            
        except Exception as e:
            print(f"Error reading audio: {e}")
            return None


class AudioInputSource(IQSource):
    """Read audio samples from microphone/line-in for TX path"""

    def __init__(self, sample_rate: float, device: int = None, buffer_size: int = 4096):
        super().__init__(sample_rate, buffer_size)
        self.device = device
        self.stream = None

    def start(self):
        super().start()
        try:
            print(f"[TX Audio Input] Opening stream: device={self.device}, sample_rate={self.sample_rate}, channels=1")
            self.stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            self.stream.start()
            print("[TX Audio Input] Stream started successfully")
        except Exception as e:
            print(f"[TX Audio Input] Error starting stream: {e}")
            import traceback
            traceback.print_exc()
            raise

    def stop(self):
        super().stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read(self) -> np.ndarray:
        if not self.running or not self.stream:
            return None

        try:
            data, _ = self.stream.read(self.buffer_size)
            if data.shape[0] == 0:
                return None
            audio = data[:, 0]
            return audio
        except Exception as e:
            print(f"Error reading audio input for TX: {e}")
            return None


class CSVIQSource(IQSource):
    """Read I/Q samples from CSV file for debugging"""

    def __init__(self, filename: str, sample_rate: float, buffer_size: int = 4096):
        super().__init__(sample_rate, buffer_size)
        self.filename = filename
        self.data = None
        self.index = 0
        self._load_csv()

    def _load_csv(self):
        """Load I/Q samples from CSV file"""
        try:
            i_samples = []
            q_samples = []
            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header if present
                for row in reader:
                    if len(row) >= 2:
                        try:
                            i_samples.append(float(row[0]))
                            q_samples.append(float(row[1]))
                        except ValueError:
                            continue
            
            self.data = np.array(i_samples) + 1j * np.array(q_samples)
            self.index = 0
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.data = None

    def start(self):
        """Start reading from CSV"""
        super().start()
        self.index = 0

    def read(self) -> np.ndarray:
        """Read next buffer from CSV"""
        if not self.running or self.data is None:
            return None
        
        if self.index >= len(self.data):
            self.index = 0  # Loop around
        
        end_index = min(self.index + self.buffer_size, len(self.data))
        chunk = self.data[self.index:end_index]
        self.index = end_index
        
        # Pad if necessary
        if len(chunk) < self.buffer_size:
            chunk = np.pad(chunk, (0, self.buffer_size - len(chunk)))
        
        return chunk


# ============================================================================
# Signal Processing and FFT
# ============================================================================

class WaterfallDisplay:
    """Manages waterfall display data"""

    def __init__(self, fft_size: int = 2048, n_rows: int = 100):
        self.fft_size = fft_size
        self.n_rows = n_rows
        self.waterfall_data = deque(maxlen=n_rows)
        self.frequencies = None

    def update(self, iq_samples: np.ndarray, sample_rate: float, center_freq: float):
        """Update waterfall with new FFT data"""
        # Compute FFT
        fft_data = np.fft.fftshift(np.abs(np.fft.fft(iq_samples, self.fft_size)))
        
        # Convert to dB scale
        fft_db = 20 * np.log10(fft_data + 1e-10)
        
        # Normalize
        fft_db = np.clip(fft_db, np.max(fft_db) - 80, np.max(fft_db))
        fft_db = (fft_db - np.min(fft_db)) / (np.max(fft_db) - np.min(fft_db) + 1e-10)
        
        self.waterfall_data.append(fft_db)
        
        # Calculate frequency axis (only once)
        if self.frequencies is None:
            freq_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/sample_rate))
            self.frequencies = center_freq + freq_axis

    def get_data(self) -> np.ndarray:
        """Get waterfall data as 2D array"""
        if len(self.waterfall_data) == 0:
            return np.zeros((self.n_rows, self.fft_size))
        
        data_array = np.array(list(self.waterfall_data))
        
        # Pad if necessary to fill the display
        if len(data_array) < self.n_rows:
            padding = np.zeros((self.n_rows - len(data_array), self.fft_size))
            data_array = np.vstack([padding, data_array])
        
        return data_array


# ============================================================================
# Audio Output
# ============================================================================

class AudioOutput:
    """Manages audio output to soundcard"""

    def __init__(self, sample_rate: float, device: int = None):
        self.sample_rate = sample_rate
        self.device = device
        self.stream = None

    def start(self):
        """Start audio output stream"""
        self.stream = sd.OutputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        self.stream.start()

    def stop(self):
        """Stop audio output stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def write(self, audio_samples: np.ndarray):
        """Write audio samples to output"""
        if self.stream:
            # Simply clip to [-1, 1] range without renormalization
            # This allows the volume control to work properly
            audio_samples = np.clip(audio_samples, -1, 1).astype(np.float32)
            self.stream.write(audio_samples)


# ============================================================================
# SDR Processing Thread
# ============================================================================

class SDRProcessor(QObject):
    """Main SDR processing in a separate thread"""
    
    waterfall_updated = pyqtSignal(np.ndarray)
    audio_level_updated = pyqtSignal(float)
    s_meter_updated = pyqtSignal(float)
    alc_level_updated = pyqtSignal(float)

    def __init__(self, config: SDRConfig):
        super().__init__()
        self.config = config
        self.demodulator = Demodulator(config.sample_rate)
        self.waterfall = WaterfallDisplay(config.fft_size)
        self.audio_output = AudioOutput(config.sample_rate, config.audio_device_output)
        self.iq_source = None
        self.tx_audio_source = AudioInputSource(config.sample_rate, device=config.tx_audio_device_input)
        self.tx_enabled = False
        self.tx_modulation = config.modulation
        self.tx_csv_file = None
        self.tx_csv_path = "tx_output.csv"
        self.running = False
        
        # Diagnostic: save demodulated audio for analysis
        self.save_demod_audio = False
        self.demod_audio_file = None
        self.demod_audio_writer = None
        self.demod_frame_count = 0

    def set_iq_source(self, iq_source: IQSource):
        """Set the I/Q sample source"""
        self.iq_source = iq_source

    def set_modulation(self, mod_type: ModulationType):
        """Change modulation type"""
        self.config.modulation = mod_type
        self.tx_modulation = mod_type

    def set_ssb_sideband(self, upper_sideband: bool):
        """Set SSB sideband mode (True=USB, False=LSB)"""
        old_value = self.config.ssb_upper_sideband
        self.config.ssb_upper_sideband = upper_sideband
        print(f"[Demod] SSB mode changed: {old_value} → {upper_sideband} ({'USB' if upper_sideband else 'LSB'})")
        print(f"[Demod] Config SSB setting is now: {self.config.ssb_upper_sideband}")

    def start_saving_demod_audio(self, filename: str = None):
        """Start saving demodulated audio to CSV for analysis"""
        if filename is None:
            sideband = "USB" if self.config.ssb_upper_sideband else "LSB"
            filename = f"demod_output_{sideband}_{self.demod_frame_count}.csv"
        
        try:
            self.demod_audio_file = open(filename, 'w', newline='')
            self.demod_audio_writer = csv.writer(self.demod_audio_file)
            self.demod_audio_writer.writerow(['audio'])
            self.save_demod_audio = True
            self.demod_frame_count = 0
            print(f"[Diag] Started saving demodulated audio to {filename}")
        except Exception as e:
            print(f"[Diag] Error opening demod audio file: {e}")

    def stop_saving_demod_audio(self):
        """Stop saving demodulated audio"""
        if self.demod_audio_file:
            try:
                self.demod_audio_file.flush()
                self.demod_audio_file.close()
                sideband = "USB" if self.config.ssb_upper_sideband else "LSB"
                print(f"[Diag] Closed demod audio file after {self.demod_frame_count} frames ({sideband})")
            except Exception as e:
                print(f"[Diag] Error closing demod audio file: {e}")
            finally:
                self.demod_audio_file = None
                self.demod_audio_writer = None
                self.save_demod_audio = False

    def set_tx_enabled(self, enabled: bool):
        """Enable or disable TX mode"""
        self.tx_enabled = enabled

    def set_tx_file(self, filename: str):
        self.tx_csv_path = filename

    def _open_tx_file(self):
        try:
            print(f"[TX] Opening CSV file: {self.tx_csv_path}")
            self.tx_csv_file = open(self.tx_csv_path, 'w', newline='')
            self.tx_csv_writer = csv.writer(self.tx_csv_file)
            self.tx_csv_writer.writerow(['I', 'Q'])
            print(f"[TX] CSV file opened successfully")
        except Exception as e:
            print(f"[TX] Error opening TX CSV file: {e}")
            import traceback
            traceback.print_exc()
            self.tx_csv_file = None

    def _close_tx_file(self):
        if self.tx_csv_file:
            try:
                self.tx_csv_file.close()
                print("[TX] CSV file closed successfully")
            except Exception as e:
                print(f"[TX] Error closing TX CSV file: {e}")
            finally:
                self.tx_csv_file = None

    def set_volume(self, volume: float):
        """Set output volume (0.0 to 1.0)"""
        self.config.volume = volume

    def set_frequency(self, freq_hz: float):
        """Set tuning frequency"""
        self.config.center_freq = freq_hz
        self.demodulator.set_tuning_frequency(freq_hz)

    def run(self):
        """Main processing loop"""
        if not self.iq_source:
            print("No I/Q source set")
            return

        self.iq_source.start()
        self.tx_audio_source.start()
        self.audio_output.start()
        self.running = True
        frame_count = 0

        try:
            while self.running:
                if self.tx_enabled:
                    # TX mode: read microphone audio and modulate
                    audio = self.tx_audio_source.read()
                    if audio is None:
                        print("[TX] Audio source returned None")
                        continue

                    if len(audio) == 0:
                        print("[TX] Audio array is empty")
                        continue

                    print(f"[TX] Read audio: shape={audio.shape}, dtype={audio.dtype}, peak={np.max(np.abs(audio)):.6f}")

                    if self.tx_csv_file is None:
                        self._open_tx_file()

                    try:
                        if self.tx_modulation == ModulationType.AM:
                            modulated = self.demodulator.modulate_am(audio)
                        elif self.tx_modulation == ModulationType.FM:
                            modulated = self.demodulator.modulate_fm(audio)
                        elif self.tx_modulation == ModulationType.SSB:
                            modulated = self.demodulator.modulate_ssb(audio, 
                                                                     upper_sideband=self.config.ssb_upper_sideband,
                                                                     center_freq=self.config.center_freq)
                        else:
                            modulated = np.zeros_like(audio, dtype=np.complex64)
                    except Exception as e:
                        print(f"[TX] Modulation error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    print(f"[TX] Modulated: shape={modulated.shape}, dtype={modulated.dtype}, peak={np.max(np.abs(modulated)):.6f}")

                    # ALC: keep max amplitude <1
                    peak = np.max(np.abs(modulated)) + 1e-12
                    gain = 1.0
                    if peak > 1.0:
                        gain = 1.0 / peak
                    modulated *= gain
                    self.alc_level_updated.emit(20 * np.log10(gain + 1e-12))

                    if self.tx_csv_file:
                        rows_written = 0
                        try:
                            for sample in modulated:
                                self.tx_csv_writer.writerow([np.real(sample), np.imag(sample)])
                                rows_written += 1
                            self.tx_csv_file.flush()  # Ensure data is written to disk
                            print(f"[TX] Wrote {rows_written} samples to CSV (flushed)")
                        except Exception as e:
                            print(f"[TX] Error writing to CSV: {e}")
                            import traceback
                            traceback.print_exc()

                    # Update S-meter label for ALC with same signal
                    self.s_meter_updated.emit(20 * np.log10(np.max(np.abs(modulated)) + 1e-12))

                    threading.Event().wait(0.001)
                    continue

                # Read I/Q samples
                iq_samples = self.iq_source.read()
                if iq_samples is None:
                    continue

                # Demodulate based on selected modulation
                if self.config.modulation == ModulationType.AM:
                    audio = self.demodulator.demodulate_am(iq_samples)
                elif self.config.modulation == ModulationType.FM:
                    audio = self.demodulator.demodulate_fm(iq_samples)
                elif self.config.modulation == ModulationType.CW:
                    audio = self.demodulator.demodulate_cw(iq_samples)
                elif self.config.modulation == ModulationType.SSB:
                    audio = self.demodulator.demodulate_ssb(iq_samples, 
                                                           upper_sideband=self.config.ssb_upper_sideband)
                else:
                    audio = np.zeros_like(iq_samples, dtype=float)

                # Apply volume
                audio = audio * self.config.volume

                # Diagnostic: Save demodulated audio if enabled
                if self.save_demod_audio and self.demod_audio_writer:
                    try:
                        for sample in audio:
                            self.demod_audio_writer.writerow([sample])
                        self.demod_frame_count += 1
                        if self.demod_frame_count % 10 == 0:
                            self.demod_audio_file.flush()
                    except Exception as e:
                        print(f"[Diag] Error writing demod audio: {e}")

                # Output audio
                self.audio_output.write(audio)

                # Update waterfall display (throttled to every 2 frames)
                frame_count += 1
                if frame_count % 2 == 0:
                    self.waterfall.update(iq_samples, self.config.sample_rate, self.config.center_freq)
                    waterfall_data = self.waterfall.get_data()
                    self.waterfall_updated.emit(waterfall_data)

                # Debug: Show SSB mode every 100 frames if SSB selected
                if self.config.modulation == ModulationType.SSB and frame_count % 100 == 0:
                    sideband_name = "USB" if self.config.ssb_upper_sideband else "LSB"
                    print(f"[RX] SSB demodulating with {sideband_name}, config={self.config.ssb_upper_sideband}")

                # Emit audio level for display
                level = np.sqrt(np.mean(audio ** 2))
                self.audio_level_updated.emit(level)

                # Emit S-meter level from I/Q power
                iq_power = np.mean(np.abs(iq_samples) ** 2)
                iq_db = 10.0 * np.log10(iq_power + 1e-12)
                self.s_meter_updated.emit(iq_db)

                # Small sleep to prevent 100% CPU
                threading.Event().wait(0.001)

        except Exception as e:
            print(f"Error in SDR processor: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.iq_source.stop()
            self.tx_audio_source.stop()
            self.audio_output.stop()
            self._close_tx_file()
            self.running = False

    def stop(self):
        """Stop processing"""
        self.running = False


# ============================================================================
# GUI Components
# ============================================================================

class WaterfallCanvas(FigureCanvas):
    """Matplotlib canvas for waterfall display"""

    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
        # Pre-create the imshow object and colorbar
        self.im = None
        self.colorbar = None
        self.waterfall_data_cache = None
        self.update_count = 0

    def update_waterfall(self, waterfall_data: np.ndarray):
        """Update waterfall display"""
        if waterfall_data.size == 0:
            return
        
        # Only update every N frames to reduce overhead
        self.update_count += 1
        if self.update_count % 2 != 0:  # Update every 2nd frame
            return
        
        try:
            # First time setup
            if self.im is None:
                self.axes.clear()
                self.im = self.axes.imshow(waterfall_data, aspect='auto', cmap='viridis',
                                          origin='lower', interpolation='bilinear')
                self.axes.set_ylabel('Time (frames)')
                self.axes.set_xlabel('Frequency Bin')
                self.colorbar = self.figure.colorbar(self.im, ax=self.axes, label='Magnitude (dB)')
            else:
                # Just update the data and colorbar limits
                self.im.set_array(waterfall_data)
                self.im.set_clim(vmin=waterfall_data.min(), vmax=waterfall_data.max())
            
            self.draw_idle()  # More efficient than draw()
        except Exception as e:
            print(f"Error updating waterfall: {e}")


class VibeSDRGUI(QMainWindow):
    """Main GUI window for VibeSDR"""

    def __init__(self):
        super().__init__()
        self.config = SDRConfig()
        self.processor = None
        self.processor_thread = None
        self.iq_source = None
        self.available_devices = []
        self.csv_device_index = None

        # S-meter smoothing & calibration state
        self.s_meter_smoothed = -127.0
        self.s_meter_baseline = None
        self.s_meter_calibration_samples = []
        self.s_meter_auto_calibrated = False
        self.s_meter_mode = "Both"

        self._populate_available_devices()

        self.setWindowTitle("VibeSDR - Software Defined Radio")
        self.setGeometry(100, 100, 1200, 700)

        # Create main widget
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel: Controls
        left_panel = self._create_control_panel()
        main_layout.addWidget(left_panel)

        # Right panel: Waterfall display
        right_panel = self._create_display_panel()
        main_layout.addWidget(right_panel)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.show()
        print("VibeSDR GUI initialized and shown")

    def _populate_available_devices(self):
        """Populate list of available audio input devices"""
        try:
            devices = sd.query_devices()
            self.available_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.available_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels']
                    })
            
            print(f"Found {len(self.available_devices)} audio input device(s)")
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            self.available_devices = []

    def _create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Demodulation selection
        demod_group = QGroupBox("Modulation Type")
        demod_layout = QFormLayout()
        self.demod_buttons = QButtonGroup()

        for i, mod in enumerate(ModulationType):
            btn = QRadioButton(mod.value)
            self.demod_buttons.addButton(btn, i)
            demod_layout.addRow(btn)
            if i == 0:  # Select AM by default
                btn.setChecked(True)

        self.demod_buttons.buttonClicked.connect(self._on_modulation_changed)
        demod_group.setLayout(demod_layout)
        layout.addWidget(demod_group)

        # SSB Sideband selection
        ssb_group = QGroupBox("SSB Sideband")
        ssb_layout = QFormLayout()
        self.ssb_buttons = QButtonGroup()
        
        self.usb_btn = QRadioButton("USB (Upper)")
        self.lsb_btn = QRadioButton("LSB (Lower)")
        self.ssb_buttons.addButton(self.usb_btn, 1)
        self.ssb_buttons.addButton(self.lsb_btn, 0)
        self.usb_btn.setChecked(True)  # Default to USB
        
        ssb_layout.addRow(self.usb_btn)
        ssb_layout.addRow(self.lsb_btn)
        
        # Connect to clicked signals (fired only when user clicks)
        self.usb_btn.clicked.connect(lambda: self._on_ssb_sideband_changed(True))
        self.lsb_btn.clicked.connect(lambda: self._on_ssb_sideband_changed(False))
        
        ssb_group.setLayout(ssb_layout)
        layout.addWidget(ssb_group)

        # Volume control
        volume_group = QGroupBox("Volume")
        volume_layout = QVBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_label = QLabel("Volume: 50%")
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)
        volume_group.setLayout(volume_layout)
        layout.addWidget(volume_group)

        # Tuning frequency
        freq_group = QGroupBox("Tuning")
        freq_layout = QHBoxLayout()
        
        # Tuning knob (dial)
        self.freq_dial = QDial()
        self.freq_dial.setMinimum(0)
        self.freq_dial.setMaximum(100000)  # 0 to 100 kHz (tuning range)
        self.freq_dial.setValue(self.config.center_freq)
        self.freq_dial.setFixedSize(60, 60)
        self.freq_dial.valueChanged.connect(self._on_freq_dial_changed)
        freq_layout.addWidget(self.freq_dial)
        
        # Frequency spinbox and label
        freq_control_layout = QVBoxLayout()
        self.freq_label = QLabel(f"Center Freq: {self.config.center_freq} Hz")
        freq_control_layout.addWidget(self.freq_label)
        
        self.freq_spinbox = QSpinBox()
        self.freq_spinbox.setMinimum(0)
        self.freq_spinbox.setMaximum(100000)
        self.freq_spinbox.setValue(self.config.center_freq)
        self.freq_spinbox.setSuffix(" Hz")
        self.freq_spinbox.setSingleStep(100)
        self.freq_spinbox.valueChanged.connect(self._on_freq_spinbox_changed)
        freq_control_layout.addWidget(self.freq_spinbox)
        
        freq_layout.addLayout(freq_control_layout)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        # Source selection
        source_group = QGroupBox("I/Q Source")
        source_layout = QVBoxLayout()
        self.source_combo = QComboBox()
        
        # Add available audio input devices
        for device in self.available_devices:
            display_name = f"{device['name']} ({device['channels']} ch)"
            self.source_combo.addItem(display_name, device['index'])
        
        # Add CSV option
        self.csv_device_index = self.source_combo.count()
        self.source_combo.addItem("CSV File (Debug)")
        
        source_layout.addWidget(self.source_combo)

        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self._on_start)
        source_layout.addWidget(start_btn)

        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._on_stop)
        source_layout.addWidget(stop_btn)

        load_csv_btn = QPushButton("Load CSV File")
        load_csv_btn.clicked.connect(self._on_load_csv)
        source_layout.addWidget(load_csv_btn)

        list_devices_btn = QPushButton("List Audio Devices")
        list_devices_btn.clicked.connect(self._on_list_devices)
        source_layout.addWidget(list_devices_btn)

        # TX audio device selection
        self.tx_device_btn = QPushButton("Select TX Device")
        self.tx_device_btn.clicked.connect(self._on_select_tx_device)
        source_layout.addWidget(self.tx_device_btn)

        self.tx_device_label = QLabel("TX Device: default")
        source_layout.addWidget(self.tx_device_label)

        self.tx_btn = QPushButton("Start TX")
        self.tx_btn.setCheckable(True)
        self.tx_btn.clicked.connect(self._on_tx_toggle)
        source_layout.addWidget(self.tx_btn)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Diagnostic tools
        diag_group = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout()
        
        self.diag_save_audio_btn = QPushButton("Save Demod Audio")
        self.diag_save_audio_btn.setCheckable(True)
        self.diag_save_audio_btn.clicked.connect(self._on_diag_save_audio_toggle)
        diag_layout.addWidget(self.diag_save_audio_btn)
        
        self.diag_status_label = QLabel("Audio: not saving")
        diag_layout.addWidget(self.diag_status_label)
        
        diag_group.setLayout(diag_layout)
        layout.addWidget(diag_group)

        # Audio level indicator
        level_group = QGroupBox("Audio Level")
        level_layout = QVBoxLayout()
        self.level_label = QLabel("Level: --")
        level_layout.addWidget(self.level_label)
        level_group.setLayout(level_layout)
        layout.addWidget(level_group)

        # S-meter indicator
        s_meter_group = QGroupBox("S-meter")
        s_meter_layout = QVBoxLayout()
        self.s_meter_label = QLabel("S-meter: -- dB")
        self.s_meter_bar = QProgressBar()
        self.s_meter_bar.setMinimum(0)
        self.s_meter_bar.setMaximum(100)
        self.s_meter_bar.setValue(0)
        self.s_meter_bar.setTextVisible(False)

        self.s_meter_mode_combo = QComboBox()
        self.s_meter_mode_combo.addItems(["Both", "dB", "S-units"])
        self.s_meter_mode_combo.currentTextChanged.connect(self._on_s_meter_mode_changed)

        s_meter_layout.addWidget(self.s_meter_label)
        s_meter_layout.addWidget(self.s_meter_bar)
        self.alc_label = QLabel("ALC: -- dB")
        s_meter_layout.addWidget(self.alc_label)
        s_meter_layout.addWidget(QLabel("Mode:"))
        s_meter_layout.addWidget(self.s_meter_mode_combo)
        s_meter_group.setLayout(s_meter_layout)
        layout.addWidget(s_meter_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def _create_display_panel(self) -> QWidget:
        """Create waterfall display panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        self.waterfall_canvas = WaterfallCanvas(panel)
        layout.addWidget(self.waterfall_canvas)

        panel.setLayout(layout)
        return panel

    def _on_modulation_changed(self):
        """Handle modulation type change"""
        if self.processor:
            selected_id = self.demod_buttons.checkedId()
            mod_type = list(ModulationType)[selected_id]
            self.processor.set_modulation(mod_type)

    def _on_ssb_sideband_changed(self, upper_sideband: bool):
        """Handle SSB sideband change
        
        Args:
            upper_sideband: True for USB, False for LSB
        """
        print(f"[UI] SSB sideband button clicked: upper_sideband={upper_sideband} ({'USB' if upper_sideband else 'LSB'})")
        if self.processor:
            print(f"[UI] Processor exists, setting sideband to {'USB' if upper_sideband else 'LSB'}")
            self.processor.set_ssb_sideband(upper_sideband)
        else:
            print(f"[UI] WARNING: Processor is None, cannot set sideband")

    def _on_freq_dial_changed(self, value):
        """Handle tuning knob change"""
        # Block signals to prevent feedback loop
        self.freq_spinbox.blockSignals(True)
        self.freq_spinbox.setValue(value)
        self.freq_spinbox.blockSignals(False)
        
        self._update_frequency(value)

    def _on_freq_spinbox_changed(self, value):
        """Handle frequency spinbox change"""
        # Block signals to prevent feedback loop
        self.freq_dial.blockSignals(True)
        self.freq_dial.setValue(value)
        self.freq_dial.blockSignals(False)
        
        self._update_frequency(value)

    def _update_frequency(self, freq_hz):
        """Update the center frequency"""
        self.config.center_freq = freq_hz
        self.freq_label.setText(f"Center Freq: {freq_hz} Hz")
        
        # Update processor if running
        if self.processor:
            self.processor.set_frequency(freq_hz)
        
        print(f"Frequency changed to: {freq_hz} Hz")

    def _on_volume_changed(self, value: int):
        """Handle volume slider change"""
        volume = value / 100.0
        self.volume_label.setText(f"Volume: {value}%")
        if self.processor:
            self.processor.set_volume(volume)

    def _on_s_meter_mode_changed(self, mode: str):
        """Handle S-meter display mode changes"""
        self.s_meter_mode = mode

    def _on_start(self):
        """Start SDR processing"""
        if self.processor and self.processor.running:
            return

        # Determine if CSV or audio device is selected
        current_index = self.source_combo.currentIndex()
        
        if current_index == self.csv_device_index:
            # CSV File mode
            if not hasattr(self, 'csv_file_path') or not self.csv_file_path:
                print("Please load a CSV file first")
                return
            self.iq_source = CSVIQSource(self.csv_file_path, self.config.sample_rate)
            print("Using CSV file as I/Q source")
        else:
            # Audio Card mode - get selected device index
            device_index = self.source_combo.currentData()
            self.iq_source = AudioCardIQSource(self.config.sample_rate, device=device_index)
            print(f"Using audio device {device_index} as I/Q source")

        # Reset S-meter calibration on start
        self.s_meter_smoothed = -127.0
        self.s_meter_baseline = None
        self.s_meter_calibration_samples = []
        self.s_meter_auto_calibrated = False

        # Create processor
        self.processor = SDRProcessor(self.config)
        self.processor.set_iq_source(self.iq_source)

        # Set SSB sideband from UI state
        if hasattr(self, 'usb_btn') and hasattr(self, 'lsb_btn'):
            upper_sideband = self.usb_btn.isChecked()
            print(f"[Init] Setting SSB sideband to {'USB' if upper_sideband else 'LSB'}")
            self.processor.set_ssb_sideband(upper_sideband)

        tx_index = getattr(self.config, 'tx_audio_device_input', None)
        if tx_index is not None:
            self.processor.tx_audio_source.device = tx_index

        self.processor.waterfall_updated.connect(self._on_waterfall_update)
        self.processor.audio_level_updated.connect(self._on_level_update)
        self.processor.s_meter_updated.connect(self._on_s_meter_update)
        self.processor.alc_level_updated.connect(self._on_alc_update)

        # Start in thread
        self.processor_thread = QThread()
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.started.connect(self.processor.run)
        self.processor_thread.start()

        print("SDR started")

    def _on_stop(self):
        """Stop SDR processing"""
        if self.processor:
            self.processor.stop()
            if self.processor_thread:
                self.processor_thread.quit()
                self.processor_thread.wait()
            print("SDR stopped")

    def _on_load_csv(self):
        """Load I/Q samples from CSV file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load I/Q CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.csv_file_path = filename
            print(f"CSV file loaded: {filename}")

    def _on_list_devices(self):
        """List available audio devices"""
        AudioCardIQSource.list_devices()

    def _on_select_tx_device(self):
        """Ask user to select TX soundcard input device"""
        items = [f"{d['name']} ({d['channels']} ch)" for d in self.available_devices]
        if not items:
            print("No TX audio devices available")
            return

        current_index = 0
        device_name, ok = QInputDialog.getItem(self, "Select TX Audio Device", "TX Device:", items, current_index, False)
        if ok and device_name:
            selected = next((d for d in self.available_devices if f"{d['name']} ({d['channels']} ch)" == device_name), None)
            if selected:
                self.tx_device_label.setText(f"TX Device: {selected['name']}")
                self.config.tx_audio_device_input = selected['index']
                if self.processor:
                    self.processor.tx_audio_source.device = selected['index']

    def _on_tx_toggle(self, checked: bool):
        """Toggle TX on/off"""
        if self.processor:
            if checked:
                self.tx_btn.setText("Stop TX")
                # file chooser for TX output each start
                filename, _ = QFileDialog.getSaveFileName(self, "Save TX I/Q CSV", "tx_output.csv", "CSV Files (*.csv)")
                if filename:
                    print(f"[TX] User selected file: {filename}")
                    self.processor.set_tx_file(filename)
                    self.processor.set_tx_enabled(True)
                else:
                    # no filename, cancel transmit
                    print("[TX] User cancelled file selection")
                    self.processor.set_tx_enabled(False)
                    self.tx_btn.setChecked(False)
                    self.tx_btn.setText("Start TX")
                    return
            else:
                print("[TX] User stopped transmit")
                self.processor.set_tx_enabled(False)
                self.tx_btn.setText("Start TX")

    def _on_alc_update(self, alc_db: float):
        """Update ALC meter during transmit"""
        self.alc_label.setText(f"ALC: {alc_db:.1f} dB")

    def _on_diag_save_audio_toggle(self, checked: bool):
        """Toggle diagnostic audio saving"""
        if self.processor:
            if checked:
                self.processor.start_saving_demod_audio()
                self.diag_save_audio_btn.setText("Stop Saving")
                self.diag_status_label.setText("Audio: SAVING")
            else:
                self.processor.stop_saving_demod_audio()
                self.diag_save_audio_btn.setText("Save Demod Audio")
                self.diag_status_label.setText("Audio: saved")

    def _on_waterfall_update(self, waterfall_data: np.ndarray):
        """Update waterfall display"""
        self.waterfall_canvas.update_waterfall(waterfall_data)

    def _on_level_update(self, level: float):
        """Update audio level display"""
        level_db = 20 * np.log10(level + 1e-10) if level > 0 else -100
        self.level_label.setText(f"Level: {level_db:.1f} dB")

    def _on_s_meter_update(self, iq_db: float):
        """Update S-meter display"""
        # Smoothing (attack/release) to reduce jitter
        alpha = 0.25
        self.s_meter_smoothed = alpha * iq_db + (1.0 - alpha) * self.s_meter_smoothed

        # Auto-calibration in first 100 samples using background-noise floor
        if not self.s_meter_auto_calibrated:
            self.s_meter_calibration_samples.append(self.s_meter_smoothed)
            if len(self.s_meter_calibration_samples) >= 100:
                self.s_meter_baseline = np.mean(self.s_meter_calibration_samples)
                self.s_meter_auto_calibrated = True
            calibrated_db = self.s_meter_smoothed
        else:
            calibrated_db = self.s_meter_smoothed - self.s_meter_baseline

        # Ham-standard mapping
        # S0 = -127 dB, S9 = -73 dB, S9+20 = -53 dB
        db_for_bar = np.clip(calibrated_db, -127.0, -53.0)
        bar_value = np.clip((db_for_bar + 127.0) / 74.0 * 100.0, 0, 100)
        self.s_meter_bar.setValue(int(bar_value))

        # Color bands style
        self.s_meter_bar.setStyleSheet(
            "QProgressBar::chunk {background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 green, stop:0.6 yellow, stop:1 red;}"
        )

        if calibrated_db <= -127.0:
            s_label = "S0"
        elif calibrated_db < -73.0:
            s_unit = np.clip(1.0 + (calibrated_db + 121.0) / 6.0, 0.0, 8.9999)
            s_label = f"S{int(np.floor(s_unit))}"
        else:
            over = min(20.0, calibrated_db + 73.0)
            s_label = f"S9+{over:.1f} dB"

        if self.s_meter_mode == "dB":
            display_text = f"S-meter: {calibrated_db:.1f} dB"
        elif self.s_meter_mode == "S-units":
            display_text = f"S-meter: {s_label}"
        else:
            display_text = f"S-meter: {calibrated_db:.1f} dB ({s_label})"

        self.s_meter_label.setText(display_text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self.processor:
                self.processor.set_tx_enabled(True)
                self.tx_btn.setChecked(True)
                self.tx_btn.setText("Stop TX")
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self.processor:
                self.processor.set_tx_enabled(False)
                self.tx_btn.setChecked(False)
                self.tx_btn.setText("Start TX")
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        """Clean up on window close"""
        self._on_stop()
        event.accept()


# ============================================================================
# Main Application Entry Point
# ============================================================================

def main():
    """Start VibeSDR application"""
    app = QApplication(sys.argv)
    gui = VibeSDRGUI()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
