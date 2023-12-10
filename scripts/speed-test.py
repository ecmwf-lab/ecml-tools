#!/usr/bin/env python3
import argparse
import random
from multiprocessing import Pool

from tqdm import tqdm

from ecml_tools.data import open_dataset


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
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to use",
        default=1,
    )

    args = parser.parse_args()

    indexes = list(range(len(open_dataset(args.path))))
    print(len(indexes))

    if args.shuffle:
        random.shuffle(indexes)

    if args.workers > 1:
        tests = Tests(
            [
                SpeedTest(args.path, indexes[i :: args.workers])
                for i in range(args.workers)
            ]
        )

        with Pool(args.workers) as pool:
            pool.map(tests, range(args.workers))
    else:
        test = SpeedTest(args.path, indexes)
        test.run()


if __name__ == "__main__":
    main()
