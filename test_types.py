import csv
from typing import TextIO, IO
try:
    print(f"TextIO[str]: {TextIO[str]}")
except Exception as e:
    print(f"TextIO[str] failed: {e}")

try:
    print(f"IO[str]: {IO[str]}")
except Exception as e:
    print(f"IO[str] failed: {e}")

try:
    print(f"csv.DictWriter[str]: {csv.DictWriter[str]}")
except Exception as e:
    print(f"csv.DictWriter[str] failed: {e}")
