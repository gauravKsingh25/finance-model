# show_active_indices.py
"""Display current active indices"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from indian_market_indices import INDIAN_INDICES

print("\n" + "="*70)
print("ACTIVE INDIAN MARKET INDICES")
print("="*70)

for i, (name, info) in enumerate(INDIAN_INDICES.items(), 1):
    print(f"{i}. {name:25s} - {info['yahoo']:25s} [{info['status']}]")

print(f"\n{'='*70}")
print(f"Total: {len(INDIAN_INDICES)} indices")
print("="*70)

print("\n📁 Streaming files:")
import os
streaming_path = "./data/streaming"
if os.path.exists(streaming_path):
    files = [f for f in os.listdir(streaming_path) if f.endswith('.parquet')]
    for f in sorted(files):
        size = os.path.getsize(os.path.join(streaming_path, f)) / 1024
        print(f"  • {f:40s} ({size:6.1f} KB)")
    print(f"\nTotal streaming files: {len(files)}")

print("\n📁 Daily files:")
daily_path = "./data/daily"
if os.path.exists(daily_path):
    files = [f for f in os.listdir(daily_path) if f.endswith('.parquet')]
    for f in sorted(files):
        size = os.path.getsize(os.path.join(daily_path, f)) / 1024
        print(f"  • {f:40s} ({size:6.1f} KB)")
    print(f"\nTotal daily files: {len(files)}")
