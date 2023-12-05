#!/usr/bin/env python3
import time
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
        "-c",
        "--count",
        help="How many item to open",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset (default is to shuffle)",
        default=True,
    )
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")

    args = parser.parse_args()

    ds = open_dataset(args.path)

    total = len(ds)
    print(f"Dataset has {total} items. Opening {args.count} items.")

    if args.shuffle:
        # take items at random
        indexes = random.sample(range(total), args.count)
    else:
        indexes = range(args.count)

    def to_human_readable(seconds):
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 60 * 60:
            return f"{seconds / 60:.2f} minutes"
        else:
            return f"{seconds / 60 / 60:.2f} hours"

    def show(start, count):
        end = time.time()
        print(
            f"Opening {count} items took {to_human_readable(end - start)} ({(end - start) / count:.4f} seconds per item)"
        )

    start = time.time()
    for i, ind in enumerate(indexes):
        ds[ind]
        if i == (args.count // 10):
            show(start, i)

        if i % 10 == 0:
            print(".", end="", flush=True)

    print()
    show(start, args.count)


if __name__ == "__main__":
    main()
