#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
import signal
import sys
import time
import zipfile
from signal import Signals

from tqdm import tqdm

from ecml_tools import __version__


def ensure_dir_exists(path):
    dirname = os.path.dirname(path)
    if not dirname:
        return
    if os.path.exists(dirname):
        return
    os.makedirs(dirname, exist_ok=True)


VERSION = "0.1"


def to_human_readable(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 60 * 60:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds / 60 / 60:.2f} hours"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "This script zips a directory dataset into a unique zip file. "
            "Mostly robust to interuptions (ctrl+C or SIGTERM) "
            "and can run incrementally."
        )
    )
    parser.add_argument("--source", help="Source argument", required=True)
    parser.add_argument("--target", help="Target argument", required=True)
    args = parser.parse_args()

    target = args.target
    tmptarget = target + ".tmp"
    if os.path.exists(target):
        print(f"❌ Target {target} already exists, exiting.")
        sys.exit(1)

    ensure_dir_exists(target)

    print(f"Listing files in {args.source}")
    start = time.time()
    files = []
    for root, _, filenames in tqdm(os.walk(args.source)):
        for filename in filenames:
            path = os.path.join(root, filename)
            if os.path.isdir(path):
                continue
            relpath = os.path.relpath(path, args.source)
            files.append(relpath)
    end = time.time()
    print(
        f"Found {len(files)} files in {args.source} (took {to_human_readable(end-start)})"
    )

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

    if not os.path.exists(tmptarget):
        print(f"Target {tmptarget} does not exist, creating it.")
        with open(tmptarget, mode="wb") as f:
            f.write("ZARR_ZIP\n".encode("utf-8"))
        zf = open_zip(tmptarget, mode="a")
        add_metadata(zf, latest_file_written=None)
        zf.close()
    else:
        print(f"Target {tmptarget} already exists, appending to it.")

    metadata = read_metadata(tmptarget)
    print(metadata)
    zf = open_zip(tmptarget, mode="a")
    already_exists = zf.namelist()
    print(f"Found {len(already_exists)} files already in {tmptarget}")

    global stop
    stop = None

    def handle_signal(signum, frame):
        signal_name = Signals(signum).name
        print(f"Received {signal_name} signal, exiting at next iteration...")
        global stop
        stop = signal_name

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGALRM, handle_signal)
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
            print(f"Closing the zip file {tmptarget} and exiting...")
            add_metadata(zf, latest_file_written=relpath)
            zf.close()
            raise

    add_metadata(zf, latest_file_written=relpath, finished=True)
    zf.close()

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    shutil.move(tmptarget, target)


if __name__ == "__main__":
    main()
