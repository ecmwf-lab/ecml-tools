import os
import sys
from collections import defaultdict

import climetlab as cml
import tqdm
import yaml

from . import Command

KEYS = ("class", "type", "stream", "expver", "levtype", "domain")


class Scan(Command):
    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--extension", default=".grib", help="Extension of the files to scan"
        )
        command_parser.add_argument(
            "--magic", help="File 'magic' to use to identify the file type. Overrides --extension"
        )
        command_parser.add_argument("paths", nargs="+", help="Paths to scan")

    def run(self, args):
        EXTENSIONS = {
            ".grib": "grib",
            ".grib1": "grib",
            ".grib2": "grib",
            ".grb": "grib",
            ".nc": "netcdf",
            ".nc4": "netcdf",
        }

        MAGICS = {'GRIB': 'grib',}

        if args.magic:
            what = MAGICS[args.magic]
            args.magic = args.magic.encode()
        else:
            what = EXTENSIONS[args.extension]

        def match(path):
            if args.magic:
                with open(path, "rb") as f:
                    return args.magic == f.read(len(args.magic))
            else:
                return path.endswith(args.extension)

        paths = []
        for path in args.paths:
            if os.path.isfile(path):

                    paths.append(path)
            else:
                for root, _, files in os.walk(path):
                    for file in files:
                        full = os.path.join(root, file)
                        paths.append(full)

        dates = set()
        gribs = defaultdict(set)
        unique = defaultdict(lambda: defaultdict(set))

        for path in tqdm.tqdm(paths, leave=False):
            if not match(path):
                continue
            for field in tqdm.tqdm(cml.load_source("file", path), leave=False):
                dates.add(field.valid_datetime())
                mars = field.as_mars()
                keys = tuple(mars.get(k) for k in KEYS)
                gribs[keys].add(path)
                for k, v in mars.items():
                    if k not in KEYS + ("date", "time", "step"):
                        unique[keys][k].add(v)

        config = dict(
            description=f"Generated by {sys.argv})",
            dataset_status="experimental",
            purpose="aifs",
            name="test",
            config_format_version=2,
            dates=dict(values=sorted(dates)),
            input=dict(join=[]),
            output=dict(
                chunking=dict(dates=1, ensembles=1),
                dtype="float32",
                flatten_grid=True,
                order_by=["valid_datetime", "param_level", "number"],
                statistics="param_level",
                statistics_end=2020,
                remapping=dict(param_level="{param}_{levelist}"),
            ),
        )

        for k, v in sorted(gribs.items()):
            request = {what: dict(path=sorted(v), **dict(zip(KEYS, k)))}
            for k, v in sorted(unique[k].items()):
                if len(v) == 1:
                    request[what][k] = list(v)[0]
                else:
                    request[what][k] = sorted(v)

            config["input"]["join"].append(request)

        with open("scan-config.yaml", "w") as f:
            print(yaml.dump(config, sort_keys=False), file=f)


command = Scan
