#!/usr/bin/env python3
"""Simple validation script for face_detection_filter.py

This script validates that the face detection filter script can be imported
and that its basic functionality works without requiring actual face detection.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_import():
    """Test that the face detection filter can be imported."""
    try:
        from face_detection_filter import FaceDetectionFilter

        print("✅ Successfully imported FaceDetectionFilter")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FaceDetectionFilter: {e}")
        return False


def test_argument_parsing():
    """Test that the script accepts command line arguments correctly."""
    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "face_detection_filter.py",
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
                result.stdout[:200] + "..."
                if len(result.stdout) > 200
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
        from face_detection_filter import FaceDetectionFilter

        # Create a temporary test instance (we won't actually process files)
        filter = FaceDetectionFilter(
            source_dir="/tmp/test_source",
            destination_dir="/tmp/test_dest",
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

        # Test that cascade files are available
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            print("✅ Haar cascade files are available")
        else:
            print("❌ Haar cascade files are not available")
            return False

        return True
    except ImportError as e:
        print(f"❌ OpenCV is not available: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔍 Validating Face Detection Filter Script")
    print("=" * 50)

    tests = [
        ("Import Test", test_import),
        ("Dependencies Test", test_dependencies),
        ("Argument Parsing Test", test_argument_parsing),
        ("Logging Test", test_logging),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        print("-" * 30)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All validation tests passed!")
        print("\n🚀 The face detection filter script is ready to use!")
        print("\n📖 Usage examples:")
        print(
            "  python face_detection_filter.py -s /Users/name/Pictures -d /Users/name/Faces_Only"
        )
        print(
            "  python face_detection_filter.py -s /path/to/source -d /path/to/destination --min-size 80"
        )
        print(
            "  python face_detection_filter.py -s /path/to/source -d /path/to/destination --dry-run"
        )
        return True
    print("❌ Some validation tests failed!")
    print("Please check the errors above and fix any issues.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
