import os
import shutil
import subprocess
import sys
from pathlib import Path

def setup_test_data(base_dir: Path):
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Create structure:
    # base/fake/a.png
    # base/fake/sub/b.jpg
    # base/real/c.png
    (base_dir / "fake" / "sub").mkdir(parents=True)
    (base_dir / "real").mkdir(parents=True)
    
    (base_dir / "fake" / "a.png").touch()
    (base_dir / "fake" / "sub" / "b.jpg").touch()
    (base_dir / "real" / "c.png").touch()
    print(f"Created test data at {base_dir}")

def verify_output(output_dir: Path):
    if not output_dir.exists():
        print("FAIL: Output directory not created")
        return False
        
    print(f"Checking output in {output_dir}")
    # We expect 'fake' and 'real' subdirs because input had them
    fake_out = output_dir / "fake"
    real_out = output_dir / "real"
    
    if not fake_out.exists() or not real_out.exists():
        print("FAIL: Output subdirectories 'fake'/'real' missing")
        return False
        
    # Check filenames
    # fake/a.png -> fake/a.png (relative to fake dir is a.png) -> a.png
    # fake/sub/b.jpg -> fake/sub_b.jpg
    
    files_fake = list(fake_out.iterdir())
    names_fake = [f.name for f in files_fake]
    print(f"Fake output files: {names_fake}")
    
    # We expect 'a.png' and 'sub_b.jpg' inside fake_out
    # Note: our logic in script:
    # relative_path = source_path.relative_to(source_root)
    # where source_root is the category dir (e.g. base/fake)
    # So base/fake/a.png -> a.png
    # base/fake/sub/b.jpg -> sub_b.jpg
    
    expected_fake = {'a.png', 'sub_b.jpg'}
    if not set(names_fake).issuperset(expected_fake):
        print(f"FAIL: Expected {expected_fake} in fake output")
        return False
        
    files_real = list(real_out.iterdir())
    names_real = [f.name for f in files_real]
    print(f"Real output files: {names_real}")
    if 'c.png' not in names_real:
        print("FAIL: c.png missing in real output")
        return False
        
    print("SUCCESS: Verification passed")
    return True

def main():
    test_dir = Path("test_sample_images_data")
    setup_test_data(test_dir)
    
    cmd = [
        "python", "sample_images.py",
        "--source_dir", str(test_dir),
        "--sample_size", "10",
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    output_dir = test_dir.parent / f"{test_dir.name}_sampled"
    try:
        if verify_output(output_dir):
            print("Cleanup...")
            shutil.rmtree(test_dir)
            shutil.rmtree(output_dir)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
