"""Automated downloader for Meta AI SAM ViT-B base checkpoint."""
import os
import urllib.request
import sys

URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEST = os.path.join(os.path.dirname(__file__), "weights", "sam_vit_b_01ec64.pth")

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading sam_vit_b_01ec64.pth: {percent}% [{count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
    sys.stdout.flush()

if __name__ == "__main__":
    if os.path.exists(DEST):
        print(f"Weights already present at {DEST}")
    else:
        print(f"Downloading SAM ViT-B weights from {URL}...")
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        urllib.request.urlretrieve(URL, DEST, reporthook=download_progress)
        print("\nDownload complete!")
