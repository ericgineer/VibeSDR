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
    QSpinBox, QDial, QProgressBar
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
        SSB demodulation: use Hilbert transform to recover modulating signal
        """
        # Apply frequency shift for tuning
        iq_samples = self.apply_frequency_shift(iq_samples)
        
        # For USB: take I, for LSB: take Q (simplified approach)
        # More complex would use Hilbert transform
        if upper_sideband:
            audio = np.real(iq_samples)
        else:
            audio = np.imag(iq_samples)
        
        # Remove DC bias
        audio = audio - np.mean(audio)
        
        # Apply low-pass filter
        audio = signal.lfilter(self.demod_filter[0], self.demod_filter[1], audio)
        
        return audio


# ============================================================================
# I/Q Sample Sources
# ============================================================================

class IQSource:
    """Base class for I/Q sample sources"""

    def __init__(self, sample_rate: float, buffer_size: int = 4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False

    def start(self):
        """Start the source"""
        self.running = True

    def stop(self):
        """Stop the source"""
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

    def __init__(self, config: SDRConfig):
        super().__init__()
        self.config = config
        self.demodulator = Demodulator(config.sample_rate)
        self.waterfall = WaterfallDisplay(config.fft_size)
        self.audio_output = AudioOutput(config.sample_rate, config.audio_device_output)
        self.iq_source = None
        self.running = False

    def set_iq_source(self, iq_source: IQSource):
        """Set the I/Q sample source"""
        self.iq_source = iq_source

    def set_modulation(self, mod_type: ModulationType):
        """Change modulation type"""
        self.config.modulation = mod_type

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
        self.audio_output.start()
        self.running = True
        frame_count = 0

        try:
            while self.running:
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
                    audio = self.demodulator.demodulate_ssb(iq_samples)
                else:
                    audio = np.zeros_like(iq_samples, dtype=float)

                # Apply volume
                audio = audio * self.config.volume

                # Output audio
                self.audio_output.write(audio)

                # Update waterfall display (throttled to every 2 frames)
                frame_count += 1
                if frame_count % 2 == 0:
                    self.waterfall.update(iq_samples, self.config.sample_rate, self.config.center_freq)
                    waterfall_data = self.waterfall.get_data()
                    self.waterfall_updated.emit(waterfall_data)

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
            self.audio_output.stop()
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

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

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
        self.processor.waterfall_updated.connect(self._on_waterfall_update)
        self.processor.audio_level_updated.connect(self._on_level_update)
        self.processor.s_meter_updated.connect(self._on_s_meter_update)

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
