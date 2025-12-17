#!/usr/bin/env python3
"""Validation script for face_detection_filter_retinaface.py

This script validates that the RetinaFace face detection filter script can be imported
and that its basic functionality works without requiring actual face detection.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_import():
    """Test that the face detection filter can be imported."""
    try:
        from face_detection_filter_retinaface import (
            FaceDetectionFilter,
            RetinaFaceDetector,
        )

        print("✅ Successfully imported FaceDetectionFilter and RetinaFaceDetector")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False


def test_argument_parsing():
    """Test that the script accepts command line arguments correctly."""
    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "face_detection_filter_retinaface.py",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("✅ Command line argument parsing works")
            print("📋 Help output preview:")
            print(
                result.stdout[:250] + "..."
                if len(result.stdout) > 250
                else result.stdout
            )
            return True
        print(f"❌ Command line argument parsing failed: {result.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error testing argument parsing: {e}")
        return False


def test_logging():
    """Test that logging is set up correctly."""
    try:
        from face_detection_filter_retinaface import FaceDetectionFilter

        # Create a temporary test instance (we won't actually process files)
        filter = FaceDetectionFilter(
            source_dir="/tmp/test_source",
            destination_dir="/tmp/test_dest",
            confidence_threshold=0.5,
            device="cpu",
            log_level="INFO",
        )

        # Test that logger exists and has handlers
        if hasattr(filter, "logger") and filter.logger.handlers:
            print("✅ Logging is properly configured")
            return True
        print("❌ Logging is not properly configured")
        return False

    except Exception as e:
        print(f"❌ Error testing logging: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    try:
        import cv2

        print(f"✅ OpenCV is available (version: {cv2.__version__})")

        import torch

        print(f"✅ PyTorch is available (version: {torch.__version__})")

        # Check MPS availability
        if torch.backends.mps.is_available():
            print(
                "✅ MPS (Metal Performance Shaders) is available for Mac M1/M2 acceleration"
            )
        else:
            print("⚠️  MPS is not available (using CPU mode)")

        # Test insightface import
        import insightface

        print("✅ InsightFace is available")

        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: uv pip install insightface opencv-python torch")
        return False


def test_retinaface_detector():
    """Test that RetinaFace detector can be initialized."""
    try:
        from face_detection_filter_retinaface import RetinaFaceDetector

        # Test CPU initialization
        detector = RetinaFaceDetector(confidence_threshold=0.5, device="cpu")
        print("✅ RetinaFaceDetector can be initialized on CPU")

        # Test auto-detection
        detector_auto = RetinaFaceDetector(confidence_threshold=0.5, device=None)
        print(
            f"✅ RetinaFaceDetector auto-detection works (using {detector_auto.device})"
        )

        return True
    except Exception as e:
        print(f"❌ Error testing RetinaFace detector: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔍 Validating RetinaFace Face Detection Filter Script")
    print("=" * 60)

    tests = [
        ("Import Test", test_import),
        ("Dependencies Test", test_dependencies),
        ("RetinaFace Detector Test", test_retinaface_detector),
        ("Argument Parsing Test", test_argument_parsing),
        ("Logging Test", test_logging),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        print("-" * 40)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All validation tests passed!")
        print("\n🚀 The RetinaFace face detection filter script is ready to use!")
        print("\n📖 Usage examples:")
        print(
            "  python face_detection_filter_retinaface.py -s /Users/name/Pictures -d /Users/name/Faces_Only"
        )
        print(
            "  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --confidence 0.7"
        )
        print(
            "  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --device mps"
        )
        print(
            "  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --dry-run"
        )
        return True
    print("❌ Some validation tests failed!")
    print("Please check the errors above and fix any issues.")
    print("\n💡 To install dependencies:")
    print("  uv pip install insightface opencv-python torch")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
