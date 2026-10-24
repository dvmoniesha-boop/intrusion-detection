# -*- coding: utf-8 -*-

"""
download_dataset.py - Download the NSL-KDD train and test files into data/.

Usage:
    python data/download_dataset.py
"""

import os
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {
    "KDDTrain+.txt": BASE_URL + "KDDTrain%2B.txt",
    "KDDTest+.txt": BASE_URL + "KDDTest%2B.txt",
}


def download():
    for name, url in FILES.items():
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest):
            print(f"   {name} already exists - skipping.")
            continue
        print(f"   Downloading {name} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        print(f"   Saved -> {dest}")


if __name__ == "__main__":
    download()
