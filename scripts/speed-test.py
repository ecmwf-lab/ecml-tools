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

    args = parser.parse_args()

    ds = open_dataset(args.path)

    indexes = list(range(len(ds)))

    if args.shuffle:
        random.shuffle(indexes)

    for i in tqdm(indexes, total=len(indexes), smoothing=0):
        ds[i]
        pass


if __name__ == "__main__":
    main()
