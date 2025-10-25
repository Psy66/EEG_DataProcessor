# 🧠 EEG Data Processing Pipeline

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![MNE](https://img.shields.io/badge/MNE-Python-orange)
![HDF5](https://img.shields.io/badge/HDF5-Data%20Storage-lightblue)
![EDA](https://img.shields.io/badge/EEG-Processing-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-GitHub-lightgrey)

**Enterprise-Grade EEG Data Processing and Dataset Preparation System**

A comprehensive, production-ready pipeline for automated processing, segmentation, and storage of EEG data from EDF files, featuring advanced signal processing and efficient HDF5 dataset management.

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Advanced Usage](#-advanced-usage)
- [Technical Documentation](#-technical-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Processing Capabilities
- **Automated EDF Processing** - Robust parsing of EEG data with comprehensive artifact handling
- **Advanced Signal Processing** - Multi-stage filtering (bandpass, notch, 3-sigma) and ICA artifact removal
- **Intelligent Segmentation** - Automated event-based segmentation with Russian-to-English label translation
- **Quality Control** - SHA256 integrity verification and file size validation
- **Channel Management** - Automatic montage application for 10/20 electrode systems

### Dataset Preparation
- **Flexible Segmentation** - Configurable segment duration and overlap settings
- **Smart Block Splitting** - Fixed-length block generation with duration validation
- **Efficient Storage** - Optimized HDF5 format with compression and chunking
- **Metadata Preservation** - Comprehensive patient demographics and recording information
- **Incremental Processing** - Resume capability with progress tracking

### Enterprise Features
- **Cloud Storage Integration** - Seamless Synology NAS integration for distributed processing
- **Parallel Processing Ready** - Modular architecture supporting high-throughput analysis
- **Production Logging** - Comprehensive audit trails and error handling
- **Configurable Pipelines** - Flexible processing parameters for different research needs
- **Data Validation** - Automated quality checks and integrity verification

## 🏗 System Architecture

```
EDF Input → Preprocessing → Segmentation → Block Generation → HDF5 Storage
     ↓            ↓             ↓              ↓               ↓
  NAS Sync    Filtering    Event-Based     Fixed-Length    Compressed
  SHA256 Check ICA Artifact Removal Label Translation  Metadata Embedding
```

### Processing Pipeline
1. **Data Acquisition** - Secure download from Synology NAS with integrity verification
2. **Signal Enhancement** - Multi-stage filtering and artifact removal pipeline
3. **Event Segmentation** - Intelligent splitting based on clinical annotations
4. **Block Generation** - Uniform duration blocks for machine learning readiness
5. **Storage Optimization** - Efficient HDF5 format with comprehensive metadata

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Synology NAS (optional, for remote storage)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/your-org/EEG_DataProcessor.git
cd EEG_DataProcessor

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## ⚡ Quick Start

### Basic Usage
```bash
python main.py
```

Select processing mode:
- **1 - Process EDFs**: Full pipeline from raw EDF to segmented data
- **2 - Prepare Dataset**: Convert segments to ML-ready HDF5 format

### Programmatic Usage
```python
from main import process_edfs, prepare_dataset

# Process individual EDF files
process_edfs(config)

# Create HDF5 datasets
prepare_dataset(config)
```

## ⚙️ Configuration

### Key Configuration Options
```yaml
processing:
  bandpass_filter: [0.5, 45]    # Frequency range for EEG analysis
  notch_filter: [50, 60]        # Power line noise removal
  segment_min_duration: 1.0     # Minimum segment length (seconds)
  block_length: 5.0            # Fixed block duration for ML
  bad_channel_threshold: 0.8    # Quality control threshold

storage:
  protocol: https              # NAS connection protocol
  overwrite_downloads: false   # Incremental processing
  file_size_limit_mb: 730      # Resource management

hdf5:
  compression: gzip           # Storage optimization
  compression_opts: 4         # Balance speed vs size
```

## 📊 Output Format

### HDF5 Dataset Structure
```python
diagnosis_segment-type.h5
├── patient_id/
│   ├── data (dataset)          # Shape: [blocks, channels, samples]
│   └── metadata/
│       ├── gender (attribute)   # Patient demographics
│       ├── age_cat (attribute)  # Age categorization
│       └── source_files (dataset) # Original file provenance
├── diagnosis (global attribute) # Clinical classification
├── segment_label (global attr)  # EEG state (Baseline, EyesOpen, etc.)
├── sampling_rate (global attr)  # Recording frequency
└── channel_names (global attr)  # Electrode configuration
```

### Supported Segment Types
- **Baseline** - Resting state EEG
- **EyesOpen/EyesClosed** - Visual activation states
- **PhoticStim** - Response to light stimulation
- **Hypervent** - Hyperventilation response
- **PostStim** - Post-stimulation recovery

## 🔧 Advanced Usage

### Custom Processing Pipeline
```python
from edf_preproc.edf_preproc import EdfPreprocessor
from edf_segmentor.EdfSegmentor import EdfSegmentor

# Initialize processors
preprocessor = EdfPreprocessor.from_config(config['processing'])
segmentor = EdfSegmentor()

# Custom processing flow
cleaned_edf = preprocessor.edf_preprocess(input_path, output_dir)
segments_csv = segmentor.create_segment_csv(input_dir, filename, output_dir)
```

### Batch Processing
```python
# Process multiple studies
studies = pd.read_csv('studies.csv')
for study in studies:
    process_single_target(config, study, api, preprocessor)
```

## 📈 Performance Optimization

### Memory Management
- Configurable file size limits
- Streaming processing for large files
- Efficient HDF5 chunking strategies

### Storage Efficiency
- GZIP compression with adjustable levels
- Intelligent metadata storage
- Incremental dataset expansion

## 🔍 Technical Documentation

### Signal Processing Chain
1. **Temporal Cropping** - Remove edge artifacts (5-second trim)
2. **Spectral Filtering** - Bandpass (0.5-45Hz) + Notch (50/60Hz)
3. **Artifact Removal** - ICA for ocular artifacts + 3-sigma outlier detection
4. **Amplitude Normalization** - Min-max scaling per channel

### Segmentation Logic
- Event annotation parsing and translation
- Exclusion of technical artifacts (stimFlash, stimAudio)
- Duration validation and quality filtering
- Automatic montage application based on channel count

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This tool is designed for research purposes. Always validate results in clinical contexts and comply with local data protection regulations.
