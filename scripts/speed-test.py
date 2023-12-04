#!/usr/bin/env python3
import argparse
import random

from tqdm import tqdm

from ecml_tools.data import open_dataset


def main():
    """Speed test for opening a dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help=(
            "Path to the dataset, use s3:// for S3 storage, "
            "http:// for HTTP storage, and /path/to/dataset for local storage"
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset",
    )
    parser.add_argument(
        "--partial",
        help="Stop after downloading a fraction of the data (ex: 0.05 for 5%)",
    )

    args = parser.parse_args()

    ds = open_dataset(args.path)

    indexes = list(range(len(ds)))
    total = len(indexes)

    if args.shuffle:
        random.shuffle(indexes)

    count = 0
    for i in tqdm(indexes, total=total, smoothing=0):
        ds[i]
        count += 1
        if count/total > float(args.partial):
            print(f"Tested {args.partial} of the data.")
            return


if __name__ == "__main__":
    main()
