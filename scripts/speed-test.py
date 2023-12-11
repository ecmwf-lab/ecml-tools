#!/usr/bin/env python3
import argparse
import random
import time
from multiprocessing import Pool

from tqdm import tqdm

from ecml_tools.data import open_dataset

VERSION = 1


def to_human_readable(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 60 * 60:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds / 60 / 60:.2f} hours"


def to_human_readable_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def show(start, count, bytes=""):
    end = time.time()
    if bytes:
        bytes = bytes / (end - start)
        bytes = "(" + to_human_readable_bytes(bytes) + "/s)"
    print(
        (
            f"Opening {count} items took {to_human_readable(end - start)}."
            f" Items per seconds: {count / (end - start) :.2f} {bytes}"
        )
    )


class SpeedTest:
    """Speed test for opening a dataset"""

    def __init__(self, path, indexes):
        self.path = path
        self.indexes = indexes

    def run(self):
        ds = open_dataset(self.path)
        print(ds)
        for i in tqdm(self.indexes):
            ds[i]
            # print(i)


class Tests:
    def __init__(self, tests):
        self.tests = tests

    def __call__(self, i):
        self.tests[i].run()


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
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset (default is to shuffle)",
        default=True,
    )
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")

    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to use",
        default=8,
    )

    args = parser.parse_args()
    nworkers = args.workers
    start = time.time()

    ds = open_dataset(args.path)

    total = len(ds)
    print(
        f"Dataset has {total} items. Opening {args.count} items using {nworkers} workers."
    )
    indexes = list(range(len(open_dataset(args.path))))

    if args.shuffle:
        # take items at random to avoid hitting the caches and use hot data
        random.shuffle(indexes)

    if args.count is not None:
        indexes = indexes[: args.count]

    if nworkers > 1:
        tests = Tests(
            [SpeedTest(args.path, indexes[i::nworkers]) for i in range(nworkers)]
        )

        with Pool(nworkers) as pool:
            pool.map(tests, range(nworkers))
    else:
        test = SpeedTest(args.path, indexes)
        test.run()

    ds = open_dataset(args.path)
    first = ds[0]
    shape, dtype, size = first.shape, first.dtype, first.size
    size_bytes = size * dtype.itemsize

    print()
    print(
        (
            f"Each item has shape {shape} and dtype {dtype}, total {size} values, "
            f"total {to_human_readable_bytes(size_bytes)}"
        )
    )
    print("Speed test version:", VERSION)
    show(start, args.count, bytes=size_bytes * args.count)


if __name__ == "__main__":
    main()
