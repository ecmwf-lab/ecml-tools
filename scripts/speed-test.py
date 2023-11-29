#!/usr/bin/env python
import argparse

from ecml_tools.data import open_dataset
from tqdm import tqdm


def main():
    """Speed test for opening a dataset """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the dataset, use s3:// for S3 storage, http:// for HTTP storage, and /path/to/dataset for local storage")
    args = parser.parse_args()

    ds = open_dataset(args.path)
    
    for i in tqdm(ds, total=len(ds), smoothing=0):
        pass
        args = parser.parse_args()
    
        ds = open_dataset(args.path)
    
        for i in tqdm(ds, total=len(ds), smoothing=0):
            pass

if __name__ == "__main__":
    main()