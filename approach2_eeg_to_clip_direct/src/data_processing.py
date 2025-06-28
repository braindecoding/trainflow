import gc
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def process_full_dataset(
    file_path, output_dir, test_size=0.2, chunk_size=50000, max_samples=None
):
    """Process FULL dataset from EP1.01.txt with memory management"""

    print("=" * 70)
    print("PROCESSING FULL MINDBIGDATA EP1.01 DATASET")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input file: {file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")

    # First pass: Count total lines
    print("\nPhase 1: Counting total lines...")
    total_lines = 0
    with open(file_path, "r") as f:
        for line in f:
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Counted {total_lines:,} lines...")

    print(f"Total lines in dataset: {total_lines:,}")

    # Second pass: Process data in chunks
    print("\nPhase 2: Processing data...")

    all_data = []
    processed_lines = 0

    with open(file_path, "r") as f:
        chunk_data = []

        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            line = line.strip()
            if not line:
                continue

            # Parse line
            parts = line.split("\t") if "\t" in line else line.split(",")

            if len(parts) >= 6:
                try:
                    subject = (
                        int(parts[4])
                        if parts[4].isdigit()
                        or (parts[4].startswith("-") and parts[4][1:].isdigit())
                        else None
                    )

                    # Filter for subjects 0-9 (corresponding to stimuli)
                    if subject is not None and 0 <= subject <= 9:
                        row = {
                            "id1": parts[0],
                            "id2": parts[1],
                            "label": parts[2],
                            "electrode": parts[3],
                            "subject": subject,
                            "trial": parts[5],
                            "eeg_data": ",".join(parts[6:]) if len(parts) > 6 else "",
                        }
                        chunk_data.append(row)

                except (ValueError, IndexError):
                    continue

            processed_lines += 1

            # Process chunk when it reaches chunk_size
            if len(chunk_data) >= chunk_size:
                all_data.extend(chunk_data)
                chunk_data = []

                # Check if we've reached max_samples
                if max_samples and len(all_data) >= max_samples:
                    print(f"Reached max samples limit: {max_samples:,}")
                    break

                # Periodic cleanup
                if len(all_data) % (chunk_size * 10) == 0:
                    print(f"  Processed {len(all_data):,} valid samples...")
                    gc.collect()

        # Add remaining data
        if chunk_data and (not max_samples or len(all_data) < max_samples):
            remaining = max_samples - len(all_data) if max_samples else len(chunk_data)
            all_data.extend(chunk_data[:remaining])

    print(f"\nPhase 3: Creating DataFrame...")
    print(f"Total valid samples (subjects 0-9): {len(all_data):,}")

    if len(all_data) == 0:
        raise ValueError("No valid data found!")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    del all_data
    gc.collect()

    print(f"DataFrame created with shape: {df.shape}")

    # Analyze data distribution
    print(f"\nData Distribution Analysis:")
    subject_counts = df["subject"].value_counts().sort_index()
    for subject, count in subject_counts.items():
        print(f"  Subject {subject}: {count:,} samples")

    # Stratified split by subject
    print(f"\nPhase 4: Creating train-test split...")

    train_dfs = []
    test_dfs = []

    for subject in range(10):
        subject_df = df[df["subject"] == subject]
        if len(subject_df) >= 2:
            train_sub, test_sub = train_test_split(
                subject_df, test_size=test_size, random_state=42
            )
            train_dfs.append(train_sub)
            test_dfs.append(test_sub)
            print(
                f"  Subject {subject}: {len(train_sub):,} train, {len(test_sub):,} test"
            )
        elif len(subject_df) == 1:
            train_dfs.append(subject_df)
            print(f"  Subject {subject}: {len(subject_df):,} train, 0 test")

    # Combine splits
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    print(f"\nFinal split:")
    print(f"  Train set: {len(train_df):,} samples")
    print(f"  Test set: {len(test_df):,} samples")

    # Save datasets
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")

    print(f"\nPhase 5: Saving datasets...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Create summary
    summary = {
        "original_total_lines": total_lines,
        "processed_lines": processed_lines,
        "valid_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "subject_distribution": subject_counts.to_dict(),
        "max_samples_limit": max_samples,
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("DATASET PROCESSING COMPLETED")
    print(f"{'='*70}")
    print(f"✓ Processed {processed_lines:,} lines from original file")
    print(f"✓ Extracted {len(df):,} valid samples (subjects 0-9)")
    print(f"✓ Train set: {len(train_df):,} samples")
    print(f"✓ Test set: {len(test_df):,} samples")
    print(f"✓ Files saved to: {output_dir}")

    return train_path, test_path, summary_path


def prepare_training_data(
    train_path, test_path, max_train_samples=None, max_test_samples=None
):
    """Prepare data for training with memory management"""

    print("Preparing training data...")

    # Load data with limits
    if max_train_samples:
        train_df = pd.read_csv(train_path, nrows=max_train_samples * 2)
        train_df = train_df.sample(
            n=min(max_train_samples, len(train_df)), random_state=42
        )
    else:
        train_df = pd.read_csv(train_path)

    if max_test_samples:
        test_df = pd.read_csv(test_path, nrows=max_test_samples * 2)
        test_df = test_df.sample(n=min(max_test_samples, len(test_df)), random_state=42)
    else:
        test_df = pd.read_csv(test_path)

    print(f"Loaded: {len(train_df):,} train, {len(test_df):,} test samples")

    def parse_eeg_data(df, name):
        eeg_signals = []
        metadata = []

        print(f"Parsing {name} EEG data...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                eeg_values = np.array(
                    [float(x.strip()) for x in row["eeg_data"].split(",") if x.strip()]
                )
                eeg_signals.append(eeg_values)

                metadata.append(
                    {
                        "electrode": row["electrode"],
                        "subject": row["subject"],
                        "trial": row["trial"],
                    }
                )
            except:
                continue

        return eeg_signals, metadata

    # Parse data
    train_signals, train_metadata = parse_eeg_data(train_df, "train")
    test_signals, test_metadata = parse_eeg_data(test_df, "test")

    # Pad signals to same length
    print("Padding signals to consistent length...")
    all_signals = train_signals + test_signals
    max_length = max(len(s) for s in all_signals)

    def pad_signals(signals, target_length):
        padded = np.zeros((len(signals), target_length))
        for i, signal in enumerate(signals):
            padded[i, : len(signal)] = signal
        return padded

    train_signals_padded = pad_signals(train_signals, max_length)
    test_signals_padded = pad_signals(test_signals, max_length)

    # Normalize signals
    print("Normalizing signals...")
    scaler = StandardScaler()
    train_signals_normalized = scaler.fit_transform(train_signals_padded)
    test_signals_normalized = scaler.transform(test_signals_padded)

    print(f"Final shapes:")
    print(f"  Train: {train_signals_normalized.shape}")
    print(f"  Test: {test_signals_normalized.shape}")
    print(f"  Signal length: {max_length}")

    return (
        train_signals_normalized,
        train_metadata,
        test_signals_normalized,
        test_metadata,
        scaler,
    )
