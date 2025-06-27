#!/usr/bin/env python3
"""
MindBigData Raw Data Loader
==========================

Proper loader for MindBigData TSV format with correct device filtering
and channel mapping for EPOC device.

Format: [id][event][device][channel][code][size][data]
Example: 67650	67636	EP	F7	7	260	4482.564102,4477.435897,4484.102564...

Key Requirements:
- Filter for EP (EPOC) device only
- Use correct EPOC 14-channel mapping
- Handle variable data lengths
- Parse TSV format correctly
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MindBigDataLoader:
    """Loader for MindBigData TSV format"""
    
    def __init__(self):
        """Initialize MindBigData loader"""
        
        # EPOC device channels (14 channels)
        self.epoc_channels = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
        
        # Device information (from official spec)
        self.devices = {
            'MW': {'name': 'MindWave', 'channels': 1, 'sfreq': 512, 'duration': 2, 'expected_size': 1024},
            'EP': {'name': 'Emotiv EPOC', 'channels': 14, 'sfreq': 128, 'duration': 2, 'expected_size': 256},
            'MU': {'name': 'Interaxon Muse', 'channels': 4, 'sfreq': 220, 'duration': 2, 'expected_size': 440},
            'IN': {'name': 'Emotiv Insight', 'channels': 5, 'sfreq': 128, 'duration': 2, 'expected_size': 256}
        }
        
        # Target device (we only want EPOC data)
        self.target_device = 'EP'
        
        logger.info("🧠 MindBigData Loader initialized")
        logger.info(f"   Target device: {self.target_device} (Emotiv EPOC)")
        logger.info(f"   Expected channels: {len(self.epoc_channels)}")
    
    def parse_tsv_line(self, line: str) -> Optional[Dict]:
        """
        Parse single TSV line
        
        Args:
            line: TSV line string
            
        Returns:
            Parsed data dictionary or None if invalid
        """
        try:
            parts = line.strip().split('\t')
            
            if len(parts) < 7:
                return None
            
            # Parse fields
            record = {
                'id': int(parts[0]),
                'event': int(parts[1]),
                'device': parts[2],
                'channel': parts[3],
                'code': int(parts[4]),
                'size': int(parts[5]),
                'data_str': parts[6]
            }
            
            # Parse data values
            data_values = []
            for val_str in record['data_str'].split(','):
                try:
                    val = float(val_str.strip())
                    data_values.append(val)
                except ValueError:
                    continue
            
            record['data'] = np.array(data_values)
            record['actual_size'] = len(data_values)
            
            return record
            
        except Exception as e:
            logger.warning(f"Failed to parse line: {e}")
            return None
    
    def load_tsv_file(self, file_path: str, max_lines: Optional[int] = None) -> List[Dict]:
        """
        Load TSV file and parse all lines
        
        Args:
            file_path: Path to TSV file
            max_lines: Maximum lines to read (for testing)
            
        Returns:
            List of parsed records
        """
        logger.info(f"📂 Loading TSV file: {file_path}")
        
        records = []
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    
                    # Skip header if present
                    if line_count == 1 and 'id' in line.lower():
                        continue
                    
                    # Parse line
                    record = self.parse_tsv_line(line)
                    if record:
                        records.append(record)
                    
                    # Progress reporting
                    if line_count % 10000 == 0:
                        logger.info(f"   Processed {line_count} lines, {len(records)} valid records")
                    
                    # Limit for testing
                    if max_lines and line_count >= max_lines:
                        break
            
            logger.info(f"✅ Loaded {len(records)} valid records from {line_count} lines")
            return records
            
        except Exception as e:
            logger.error(f"❌ Failed to load file: {e}")
            return []
    
    def filter_epoc_data(self, records: List[Dict]) -> List[Dict]:
        """
        Filter for EPOC device data only
        
        Args:
            records: All parsed records
            
        Returns:
            EPOC-only records
        """
        logger.info(f"🔍 Filtering for {self.target_device} (EPOC) device...")
        
        epoc_records = []
        device_counts = {}
        
        for record in records:
            device = record['device']
            
            # Count devices
            device_counts[device] = device_counts.get(device, 0) + 1
            
            # Filter for EPOC
            if device == self.target_device:
                epoc_records.append(record)
        
        logger.info(f"📊 Device distribution:")
        for device, count in device_counts.items():
            logger.info(f"   {device}: {count} records")
        
        logger.info(f"✅ Filtered to {len(epoc_records)} EPOC records")
        return epoc_records
    
    def organize_by_trials(self, epoc_records: List[Dict]) -> Dict:
        """
        Organize EPOC records by trials (events)
        
        Args:
            epoc_records: EPOC device records
            
        Returns:
            Dictionary organized by event/trial
        """
        logger.info("🔄 Organizing data by trials...")
        
        trials = {}
        channel_counts = {}
        code_counts = {}
        
        for record in epoc_records:
            event = record['event']
            channel = record['channel']
            code = record['code']
            
            # Count channels and codes
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
            code_counts[code] = code_counts.get(code, 0) + 1
            
            # Organize by trial
            if event not in trials:
                trials[event] = {}
            
            trials[event][channel] = {
                'code': code,
                'data': record['data'],
                'size': record['actual_size']
            }
        
        logger.info(f"📊 Data organization:")
        logger.info(f"   Total trials: {len(trials)}")
        logger.info(f"   Channels found: {list(channel_counts.keys())}")
        logger.info(f"   Digit codes: {sorted(code_counts.keys())}")
        
        # Check channel coverage
        missing_channels = set(self.epoc_channels) - set(channel_counts.keys())
        extra_channels = set(channel_counts.keys()) - set(self.epoc_channels)
        
        if missing_channels:
            logger.warning(f"⚠️ Missing EPOC channels: {missing_channels}")
        if extra_channels:
            logger.warning(f"⚠️ Extra channels found: {extra_channels}")
        
        return trials
    
    def create_trial_matrix(self, trial_data: Dict, target_length: int = 256) -> Tuple[np.ndarray, int]:
        """
        Create trial matrix from channel data
        
        Args:
            trial_data: Trial data dictionary
            target_length: Target time points
            
        Returns:
            Trial matrix (n_channels, n_timepoints) and digit code
        """
        # Get digit code (should be same for all channels in trial)
        codes = [ch_data['code'] for ch_data in trial_data.values()]
        if len(set(codes)) > 1:
            logger.warning(f"Multiple codes in trial: {codes}")
        digit_code = codes[0] if codes else -1
        
        # Create matrix
        trial_matrix = np.zeros((len(self.epoc_channels), target_length))
        
        for ch_idx, channel in enumerate(self.epoc_channels):
            if channel in trial_data:
                data = trial_data[channel]['data']
                
                # Resample/pad to target length
                if len(data) >= target_length:
                    # Downsample
                    indices = np.linspace(0, len(data)-1, target_length, dtype=int)
                    trial_matrix[ch_idx] = data[indices]
                else:
                    # Pad with zeros
                    trial_matrix[ch_idx, :len(data)] = data
            # else: channel missing, leave as zeros
        
        return trial_matrix, digit_code
    
    def load_dataset(self, file_path: str, max_trials: Optional[int] = None,
                    target_length: int = 256) -> Dict:
        """
        Load complete dataset from TSV file
        
        Args:
            file_path: Path to TSV file
            max_trials: Maximum trials to load
            target_length: Target time points per trial
            
        Returns:
            Dataset dictionary
        """
        logger.info("🚀 Loading MindBigData dataset...")
        
        # Load and parse TSV
        records = self.load_tsv_file(file_path)
        if not records:
            return {}
        
        # Filter for EPOC
        epoc_records = self.filter_epoc_data(records)
        if not epoc_records:
            logger.error("❌ No EPOC records found!")
            return {}
        
        # Organize by trials
        trials = self.organize_by_trials(epoc_records)
        if not trials:
            logger.error("❌ No trials organized!")
            return {}
        
        # Convert to matrices
        logger.info("🔄 Converting to trial matrices...")
        
        trial_matrices = []
        trial_labels = []
        valid_trials = 0
        
        for event_id, trial_data in trials.items():
            # Check if trial has enough channels
            available_channels = set(trial_data.keys()) & set(self.epoc_channels)
            
            if len(available_channels) >= 1:  # At least 1 channel for testing
                trial_matrix, digit_code = self.create_trial_matrix(trial_data, target_length)
                
                if digit_code >= 0:  # Valid digit code
                    trial_matrices.append(trial_matrix)
                    trial_labels.append(digit_code)
                    valid_trials += 1
                    
                    if max_trials and valid_trials >= max_trials:
                        break
            
            if valid_trials % 100 == 0 and valid_trials > 0:
                logger.info(f"   Processed {valid_trials} valid trials")
        
        # Convert to arrays
        if trial_matrices:
            eeg_data = np.array(trial_matrices)
            labels = np.array(trial_labels)
            
            logger.info(f"✅ Dataset loaded successfully:")
            logger.info(f"   EEG data shape: {eeg_data.shape}")
            logger.info(f"   Labels shape: {labels.shape}")
            logger.info(f"   Digit distribution: {np.bincount(labels)}")
            
            return {
                'eeg_data': eeg_data,
                'labels': labels,
                'channels': self.epoc_channels,
                'sampling_rate': 128,
                'signal_duration': 2.0,  # 2 seconds per signal
                'n_trials': len(trial_matrices),
                'n_channels': len(self.epoc_channels),
                'n_timepoints': target_length
            }
        else:
            logger.error("❌ No valid trials found!")
            return {}


def test_loader():
    """Test the MindBigData loader with real data"""
    print("🧪 Testing MindBigData Loader with REAL DATA...")

    # Use real MindBigData file
    real_tsv_path = r"d:\trainflow\dataset\datasets\EP1.01.txt"

    if not os.path.exists(real_tsv_path):
        print(f"❌ Real data file not found: {real_tsv_path}")
        print("🔄 Creating synthetic test data...")
        # Create synthetic TSV data as fallback
        synthetic_tsv = """id	event	device	channel	code	size	data
1	1	EP	AF3	5	256	100.1,101.2,99.8,102.1,98.9,101.5,100.3,99.7,102.0,101.1,100.5,99.2,101.8,100.7,99.4,102.3,101.0,100.8,99.6,101.4,100.2,99.9,101.7,100.6,99.3,102.2,101.3,100.4,99.8,101.6,100.9,99.5,102.4,101.2,100.7,99.1,101.9,100.8,99.7,102.0,101.4,100.3,99.6,101.5,100.1,99.8,102.1,101.0,100.5,99.4,101.8,100.7,99.2,102.3,101.1,100.6,99.9,101.7,100.4,99.3,102.2,101.3,100.8,99.5,101.6,100.9,99.7,102.4,101.2,100.1,99.8,101.9,100.5,99.4,102.0,101.4,100.3,99.6,101.5,100.7,99.2,102.1,101.0,100.8,99.9,101.8,100.6,99.3,102.3,101.1,100.4,99.5,101.7,100.9,99.7,102.2,101.3,100.2,99.8,101.6,100.5,99.4,102.4,101.2,100.7,99.1,101.9,100.8,99.6,102.0,101.4,100.3,99.9,101.5,100.1,99.8,102.1,101.0,100.6,99.2,101.8,100.7,99.5,102.3,101.1,100.4,99.3,101.7,100.5,99.8,102.0,101.2,100.7,99.4,101.9,100.6,99.1,102.3,101.4,100.3,99.5,101.8,100.9,99.7,102.1,101.0,100.8,99.2,101.6,100.4,99.9,102.2,101.3,100.5,99.6,101.7,100.1,99.8,102.4,101.1,100.7,99.3,101.5,100.9,99.4,102.0,101.2,100.6,99.8,101.8,100.3,99.5,102.3,101.4,100.1,99.7,101.9,100.8,99.2,102.1,101.0,100.5,99.9,101.6,100.4,99.6,102.2,101.3,100.7,99.1,101.7,100.9,99.8,102.4,101.2,100.3,99.4,101.5,100.6,99.5,102.0,101.1,100.8,99.7,101.8,100.2,99.3,102.3,101.4,100.9,99.9,101.9,100.5,99.6,102.1,101.0,100.7,99.8,101.6,100.4,99.2,102.2,101.3,100.1,99.5,101.7,100.8,99.4,102.4,101.2,100.6,99.7,101.5,100.3,99.9,102.0,101.1,100.9,99.1,101.8,100.5,99.8,102.3,101.4,100.2,99.6,101.9,100.7,99.3,102.1,101.0,100.8,99.5,101.6,100.4,99.7,102.2,101.3,100.1,99.9,101.7,100.6,99.2,102.4,101.2,100.5,99.4,101.5,100.9,99.8,102.0,101.1,100.3,99.6,101.8,100.7,99.1
2	1	EP	F7	5	256	200.1,201.2,199.8,202.1,198.9,201.5,200.3,199.7,202.0,201.1,200.5,199.2,201.8,200.7,199.4,202.3,201.0,200.8,199.6,201.4,200.2,199.9,201.7,200.6,199.3,202.2,201.3,200.4,199.8,201.6,200.9,199.5,202.4,201.2,200.7,199.1,201.9,200.8,199.7,202.0,201.4,200.3,199.6,201.5,200.1,199.8,202.1,201.0,200.5,199.4,201.8,200.7,199.2,202.3,201.1,200.6,199.9,201.7,200.4,199.3,202.2,201.3,200.8,199.5,201.6,200.9,199.7,202.4,201.2,200.1,199.8,201.9,200.5,199.4,202.0,201.4,200.3,199.6,201.5,200.7,199.2,202.1,201.0,200.8,199.9,201.8,200.6,199.3,202.3,201.1,200.4,199.5,201.7,200.9,199.7,202.2,201.3,200.2,199.8,201.6,200.5,199.4,202.4,201.2,200.7,199.1,201.9,200.8,199.6,202.0,201.4,200.3,199.9,201.5,200.1,199.8,202.1,201.0,200.6,199.2,201.8,200.7,199.5,202.3,201.1,200.4,199.3,201.7,200.5,199.8,202.0,201.2,200.7,199.4,201.9,200.6,199.1,202.3,201.4,200.3,199.5,201.8,200.9,199.7,202.1,201.0,200.8,199.2,201.6,200.4,199.9,202.2,201.3,200.5,199.6,201.7,200.1,199.8,202.4,201.1,200.7,199.3,201.5,200.9,199.4,202.0,201.2,200.6,199.8,201.8,200.3,199.5,202.3,201.4,200.1,199.7,201.9,200.8,199.2,202.1,201.0,200.5,199.9,201.6,200.4,199.6,202.2,201.3,200.7,199.1,201.7,200.9,199.8,202.4,201.2,200.3,199.4,201.5,200.6,199.5,202.0,201.1,200.8,199.7,201.8,200.2,199.3,202.3,201.4,200.9,199.9,201.9,200.5,199.6,202.1,201.0,200.7,199.8,201.6,200.4,199.2,202.2,201.3,200.1,199.5,201.7,200.8,199.4,202.4,201.2,200.6,199.7,201.5,200.3,199.9,202.0,201.1,200.9,199.1,201.8,200.5,199.8,202.3,201.4,200.2,199.6,201.9,200.7,199.3,202.1,201.0,200.8,199.5,201.6,200.4,199.7,202.2,201.3,200.1,199.9,201.7,200.6,199.2,202.4,201.2,200.5,199.4,201.5,200.9,199.8,202.0,201.1,200.3,199.6,201.8,200.7,199.1
3	2	EP	AF3	7	256	150.1,151.2,149.8,152.1,148.9,151.5,150.3,149.7,152.0,151.1,150.5,149.2,151.8,150.7,149.4,152.3,151.0,150.8,149.6,151.4,150.2,149.9,151.7,150.6,149.3,152.2,151.3,150.4,149.8,151.6,150.9,149.5,152.4,151.2,150.7,149.1,151.9,150.8,149.7,152.0,151.4,150.3,149.6,151.5,150.1,149.8,152.1,151.0,150.5,149.4,151.8,150.7,149.2,152.3,151.1,150.6,149.9,151.7,150.4,149.3,152.2,151.3,150.8,149.5,151.6,150.9,149.7,152.4,151.2,150.1,149.8,151.9,150.5,149.4,152.0,151.4,150.3,149.6,151.5,150.7,149.2,152.1,151.0,150.8,149.9,151.8,150.6,149.3,152.3,151.1,150.4,149.5,151.7,150.9,149.7,152.2,151.3,150.2,149.8,151.6,150.5,149.4,152.4,151.2,150.7,149.1,151.9,150.8,149.6,152.0,151.4,150.3,149.9,151.5,150.1,149.8,152.1,151.0,150.6,149.2,151.8,150.7,149.5,152.3,151.1,150.4,149.3,151.7,150.5,149.8,152.0,151.2,150.7,149.4,151.9,150.6,149.1,152.3,151.4,150.3,149.5,151.8,150.9,149.7,152.1,151.0,150.8,149.2,151.6,150.4,149.9,152.2,151.3,150.5,149.6,151.7,150.1,149.8,152.4,151.1,150.7,149.3,151.5,150.9,149.4,152.0,151.2,150.6,149.8,151.8,150.3,149.5,152.3,151.4,150.1,149.7,151.9,150.8,149.2,152.1,151.0,150.5,149.9,151.6,150.4,149.6,152.2,151.3,150.7,149.1,151.7,150.9,149.8,152.4,151.2,150.3,149.4,151.5,150.6,149.5,152.0,151.1,150.8,149.7,151.8,150.2,149.3,152.3,151.4,150.9,149.9,151.9,150.5,149.6,152.1,151.0,150.7,149.8,151.6,150.4,149.2,152.2,151.3,150.1,149.5,151.7,150.8,149.4,152.4,151.2,150.6,149.7,151.5,150.3,149.9,152.0,151.1,150.9,149.1,151.8,150.5,149.8,152.3,151.4,150.2,149.6,151.9,150.7,149.3,152.1,151.0,150.8,149.5,151.6,150.4,149.7,152.2,151.3,150.1,149.9,151.7,150.6,149.2,152.4,151.2,150.5,149.4,151.5,150.9,149.8,152.0,151.1,150.3,149.6,151.8,150.7,149.1"""
    
        # Save synthetic data
        test_file = "test_mindbigdata.tsv"
        with open(test_file, 'w') as f:
            f.write(synthetic_tsv)

        # Test loader with synthetic data
        loader = MindBigDataLoader()
        dataset = loader.load_dataset(test_file, max_trials=10)

        # Cleanup
        os.remove(test_file)
    else:
        # Test loader with real data
        print(f"✅ Found real data file: {real_tsv_path}")
        loader = MindBigDataLoader()
        dataset = loader.load_dataset(real_tsv_path, max_trials=100)  # Limit for testing
    
    if dataset:
        print(f"✅ Test completed:")
        print(f"   EEG data: {dataset['eeg_data'].shape}")
        print(f"   Labels: {dataset['labels'].shape}")
        print(f"   Channels: {len(dataset['channels'])}")
        print(f"   Sampling rate: {dataset['sampling_rate']}")
    else:
        print("❌ Test failed")


if __name__ == "__main__":
    test_loader()
