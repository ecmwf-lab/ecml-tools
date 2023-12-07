#!/usr/bin/env python3
import zipfile
from tqdm import tqdm
import json
import glob
import datetime
import os
import signal
from signal import Signals
import sys

import argparse

from ecml_tools import __version__


def ensure_dir_exists(path):
    dirname = os.path.dirname(path)
    if not dirname:
        return
    if os.path.exists(dirname):
        return
    os.makedirs(dirname, exist_ok=True)


VERSION = "0.1"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "This script zips a zarr dataset into a unique zip file. "
            "Mostly robust to polite interuptions (ctrl+C SIGTERM) "
            "and can run incrementally."
        )
    )
    parser.add_argument("--source", help="Source argument", required=True)
    parser.add_argument("--target", help="Target argument", required=True)
    args = parser.parse_args()

    assert args.source.endswith(".zarr")
    assert args.target.endswith(".zip")

    ensure_dir_exists(args.target)

    # list all files in the source directory
    files = []
    for root, _, filenames in os.walk(args.source):
        for filename in filenames:
            path = os.path.join(root, filename)
            if os.path.isdir(path):
                continue
            relpath = os.path.relpath(path, args.source)
            files.append(relpath)
    files = sorted(files)
    print(f"Found {len(files)} files in {args.source}")

    def open_zip(path, mode):
        return zipfile.ZipFile(
            path, mode=mode, compression=zipfile.ZIP_STORED, allowZip64=True
        )

    def read_metadata(path):
        zf = open_zip(path, mode="r")
        return json.loads(zf.comment.decode("utf-8"))

    def add_metadata(zf, latest_file_written, finished=False):
        comment = json.dumps(
            {
                "ecml_tool": __version__,
                "version": VERSION,
                "latest_file_written": latest_file_written,
                "latest_write_timestamp": datetime.datetime.utcnow().isoformat(),
                "finished": finished,
            },
        )
        assert len(comment) < 65536, "Comment is too long for zip"
        zf.comment = comment.encode("utf-8")
        zf.close()

    if not os.path.exists(args.target):
        print(f"Target {args.target} does not exist, creating it.")
        with open(args.target, mode="wb") as f:
            f.write("ZARR_ZIP\n".encode("utf-8"))
        zf = open_zip(args.target, mode="a")
        add_metadata(zf, latest_file_written=None)
        zf.close()

    metadata = read_metadata(args.target)
    print(metadata)
    zf = open_zip(args.target, mode="a")
    already_exists = zf.namelist()
    print(f"Found {len(already_exists)} files already in {args.target}")
    print(os.getpid())

    global stop
    stop = None

    def handle_signal(signum, frame):
        signal_name = Signals(signum).name
        print(f"Received {signal_name} signal, exiting at next iteration...")
        global stop
        stop = signal_name

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for relpath in tqdm(files):
        if relpath in already_exists:
            continue
        path = os.path.join(args.source, relpath)
        with open(path, "rb") as f:
            data = f.read()
        assert isinstance(data, bytes)

        try:
            zf.writestr(relpath, data)
            if stop is not None:
                raise Exception(f"Received {stop} signal")
        except Exception as e:
            print(f"An error occurred while writing to the zip file: {e}")
            print(f"Closing the zip file {args.target} and exiting...")
            add_metadata(zf, latest_file_written=relpath)
            zf.close()
            raise

    add_metadata(zf, latest_file_written=relpath, finished=True)
    zf.close()

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    main()
