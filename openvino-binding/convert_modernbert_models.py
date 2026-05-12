#!/usr/bin/env python3
"""
Convert ModernBERT classification and PII models from HuggingFace to OpenVINO IR format
"""

import os
import sys
import shutil
from pathlib import Path

try:
    import openvino as ov

    print(f"✓ OpenVINO imported: {ov.__version__}")
except ImportError:
    print("✗ OpenVINO not installed. Install with: pip install openvino")
    sys.exit(1)

try:
    from transformers import (
        AutoTokenizer,
        AutoConfig,
    )
    import torch

    print("✓ Transformers and PyTorch imported")
except ImportError:
    print(
        "✗ Transformers/PyTorch not installed. Install with: pip install transformers torch"
    )
    sys.exit(1)

try:
    from optimum.intel import OVModelForSequenceClassification, OVModelForTokenClassification

    print("✓ optimum-intel imported")
except ImportError:
    print("✗ optimum-intel not installed. Install with: pip install optimum[openvino]")
    sys.exit(1)

# Model paths in the semantic-router models directory
MODELS_DIR = Path("../models")
OUTPUT_BASE_DIR = Path("./test_models")

# Models to convert
MODELS_TO_CONVERT = [
    {
        "name": "category_classifier",
        "path": MODELS_DIR / "category_classifier_modernbert-base_model",
        "output": OUTPUT_BASE_DIR / "category_classifier_modernbert",
        "type": "sequence_classification",
        "description": "ModernBERT Category Classifier",
    },
    {
        "name": "jailbreak_classifier",
        "path": MODELS_DIR / "jailbreak_classifier_modernbert-base_model",
        "output": OUTPUT_BASE_DIR / "jailbreak_classifier_modernbert",
        "type": "sequence_classification",
        "description": "ModernBERT Jailbreak Classifier",
    },
    {
        "name": "pii_classifier",
        "path": MODELS_DIR / "pii_classifier_modernbert-base_model",
        "output": OUTPUT_BASE_DIR / "pii_classifier_modernbert",
        "type": "sequence_classification",
        "description": "ModernBERT PII Sequence Classifier",
    },
    {
        "name": "pii_token_classifier",
        "path": MODELS_DIR / "pii_classifier_modernbert-base_presidio_token_model",
        "output": OUTPUT_BASE_DIR / "pii_token_classifier_modernbert",
        "type": "token_classification",
        "description": "ModernBERT PII Token Classifier (Presidio)",
    },
]


def convert_model(model_info):
    """Convert a single model to OpenVINO IR format"""
    model_path = model_info["path"]
    output_dir = model_info["output"]
    model_type = model_info["type"]
    description = model_info["description"]

    print(f"\n{'='*70}")
    print(f"Converting: {description}")
    print(f"Source: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Type: {model_type}")
    print(f"{'='*70}")

    # Check if model exists
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return False

    # Check if already converted
    if (output_dir / "openvino_model.xml").exists():
        print(f"✓ Model already converted")
        return True

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load config to check model type and get num_labels
        config = AutoConfig.from_pretrained(model_path)
        num_labels = getattr(config, "num_labels", 2)
        print(f"  Model config: num_labels={num_labels}")

        # Use optimum-intel OVModel* classes — proper HuggingFace → OpenVINO export
        print("  Converting to OpenVINO IR format via optimum-intel...")
        if model_type == "sequence_classification":
            ov_model = OVModelForSequenceClassification.from_pretrained(
                model_path, export=True
            )
        elif model_type == "token_classification":
            ov_model = OVModelForTokenClassification.from_pretrained(
                model_path, export=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"✓ Model converted from {model_path}")

        # Save in OpenVINO IR format
        ov_model.save_pretrained(output_dir)
        print(f"✓ OpenVINO model saved")

        # Save tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer saved")

        # Test inference
        print(f"\n  Testing inference...")
        test_result = ov_model(
            **tokenizer(
                "Test inference",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
        )
        print(f"  ✓ Inference test passed: logits shape = {test_result.logits.shape}")

        # List output files
        print(f"  Output files:")
        for f in sorted(output_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.0f} KB)")

        print(f"\n✓ Successfully converted: {description}")
        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print(f"{'='*70}")
    print(f"ModernBERT Models to OpenVINO Converter")
    print(f"{'='*70}")
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_BASE_DIR.absolute()}")
    print(f"Number of models to convert: {len(MODELS_TO_CONVERT)}")

    # Create output directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Convert each model
    results = {}
    for model_info in MODELS_TO_CONVERT:
        success = convert_model(model_info)
        results[model_info["name"]] = success

    # Summary
    print(f"\n{'='*70}")
    print(f"Conversion Summary")
    print(f"{'='*70}")

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print(f"\nTotal: {successful}/{total} models converted successfully")

    if successful == total:
        print(f"\n✓ All models ready for OpenVINO binding tests!")
    elif successful > 0:
        print(f"\n⚠️  Some models converted, others may not be available")
    else:
        print(f"\n✗ No models converted successfully")
        sys.exit(1)

    print(f"\nTo use these models in Go:")
    print(
        f"  - Category Classifier: {OUTPUT_BASE_DIR}/category_classifier_modernbert/openvino_model.xml"
    )
    print(
        f"  - Jailbreak Classifier: {OUTPUT_BASE_DIR}/jailbreak_classifier_modernbert/openvino_model.xml"
    )
    print(
        f"  - PII Classifier: {OUTPUT_BASE_DIR}/pii_classifier_modernbert/openvino_model.xml"
    )
    print(
        f"  - PII Token Classifier: {OUTPUT_BASE_DIR}/pii_token_classifier_modernbert/openvino_model.xml"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
