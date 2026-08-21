"""Dataset metadata, descriptions, and statistics."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "data" / "datasets"

DATASET_DESCRIPTIONS = {
    "massbench_adenocarcinoma": {
        "title": "MassBench Adenocarcinoma",
        "description": "Mass spectrometry dataset for adenocarcinoma classification with batch effects.",
        "task": "Multi-class classification of sample types in adenocarcinoma specimens",
    },
    "massbench_alzheimer": {
        "title": "MassBench Alzheimer",
        "description": "Mass spectrometry proteomics data for Alzheimer's disease classification across multiple diagnostic groups.",
        "task": "Multi-class classification of cognitive status and disease type",
    },
    "massbench_benchmark": {
        "title": "MassBench Benchmark",
        "description": "Standardized mass spectrometry benchmark dataset with samples from multiple protocols and preparation methods.",
        "task": "Multi-class classification across different experimental techniques",
    },
}

# Static metadata (precomputed from data exploration)
DATASET_METADATA = {
    "massbench_adenocarcinoma": {
        "train_samples": 434,
        "test_samples": 208,
        "train_features": 6464,
        "test_features": 6463,
        "classes": {"1": 336, "QC": 50, "0": 48},
        "train_classes": 3,
        "test_classes": 3,
        "train_batches": 2,
        "test_batches": 1,
        "train_batch_info": {"1": 217, "2": 217},
        "test_batch_info": {"3": 208},
        "domain": "Mass Spectrometry",
    },
    "massbench_alzheimer": {
        "train_samples": 768,
        "test_samples": 211,
        "train_features": 899,
        "test_features": 898,
        "classes": {
            "MCI-AD": 162,
            "CU": 142,
            "MCI-other": 132,
            "DEM-AD": 126,
            "NPH": 96,
            "pool": 64,
            "DEM-other": 42,
        },
        "train_classes": 7,
        "test_classes": 6,
        "train_batches": 16,
        "test_batches": 6,
        "train_batch_info": {
            "Batch-05": 48, "Batch-06": 48, "Batch-07": 48, "Batch-08": 48,
            "Batch-09": 48, "Batch-10": 48, "Batch-11": 48, "Batch-12": 48,
            "Batch-14": 48, "Batch-15": 48, "Batch-16": 48, "Batch-17": 48,
            "Batch-18": 48, "Batch-19": 48, "Batch-20": 48, "Batch-21": 48
        },
        "test_batch_info": {
            "Batch-00": 2, "Batch-03": 48, "Batch-04": 48, "Batch-13": 47,
            "Batch-22": 38, "Batch-23": 28
        },
        "domain": "Proteomics / Mass Spectrometry",
    },
    "massbench_benchmark": {
        "train_samples": 749,
        "test_samples": 278,
        "train_features": 1238,
        "test_features": 1237,
        "classes": {"Bio": 141, "SRM": 129, "AA": 123, "FA": 122, "PP": 120, "Full": 114},
        "train_classes": 6,
        "test_classes": 6,
        "train_batches": 5,
        "test_batches": 2,
        "train_batch_info": {
            "Batch 1": 144, "Batch 2": 156, "Batch 3": 144, "Batch 4": 149, "Batch 7": 156
        },
        "test_batch_info": {
            "Batch 5": 143, "Batch 6": 135
        },
        "domain": "Mass Spectrometry (Multi-Protocol)",
    },
}


def get_dataset_info_markdown(dataset_key: str) -> str:
    """Generate markdown displaying dataset information and metrics."""
    if dataset_key not in DATASET_DESCRIPTIONS:
        return "Unknown dataset"
    
    desc = DATASET_DESCRIPTIONS[dataset_key]
    meta = DATASET_METADATA.get(dataset_key, {})
    
    # Calculate basic statistics
    total_train = meta.get("train_samples", 0)
    total_test = meta.get("test_samples", 0)
    total_samples = total_train + total_test
    class_balance = meta.get("classes", {})
    
    # Format class information
    class_info = []
    if class_balance:
        for cls_name, count in sorted(
            class_balance.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (count / total_train * 100) if total_train > 0 else 0
            class_info.append(f"  - {cls_name}: {count} samples ({pct:.1f}%)")
    
    class_section = "\n".join(class_info) if class_info else "N/A"
    
    # Format batch information
    batch_info = []
    train_batches = meta.get("train_batch_info", {})
    test_batches = meta.get("test_batch_info", {})
    
    for b_name, count in sorted(train_batches.items()):
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        batch_info.append(f"  - batch {b_name}: {count} samples ({pct:.1f}%; train)")
    
    for b_name, count in sorted(test_batches.items()):
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        batch_info.append(f"  - batch {b_name}: {count} samples ({pct:.1f}%; test)")
        
    batch_section = "\n".join(batch_info) if batch_info else "N/A"

    # Build markdown
    md = f"""
### {desc['title']}

**Description:**  
{desc['description']}

**Task:**  
{desc['task']}

---

**Dataset Metrics:**

| Metric | Value |
|--------|-------|
| Total Samples (Train + Test) | {total_samples:,} |
| Training Set | {total_train:,} samples |
| Test Set | {total_test:,} |
| Features (Train) | {meta.get('train_features', 'N/A'):,} |
| Features (Test) | {meta.get('test_features', 'N/A'):,} |
| Classes (Train) | {meta.get('train_classes', 'N/A')} |
| Classes (Test) | {meta.get('test_classes', 'N/A')} |
| Batches (Train) | {meta.get('train_batches', 'N/A')} |
| Batches (Test) | {meta.get('test_batches', 'N/A')} |
| Domain | {meta.get('domain', 'N/A')} |

**Class Distribution (Training Set):**

{class_section}

**Batch Distribution (Training + Test sets):**

{batch_section}

---

**Submission Instructions:**
1. Train your model on the training split
2. Predict on the test split
3. Submit your corrected data and model code
4. Evaluation runs server-side on private labels
"""
    return md


def get_dataset_summary(dataset_key: str) -> dict:
    """Get structured dataset metadata."""
    if dataset_key not in DATASET_METADATA:
        return {}
    return DATASET_METADATA[dataset_key]
