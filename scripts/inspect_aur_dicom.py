#!/usr/bin/env python3
"""Inspect only explicitly supplied files; print anonymous metadata counts."""

import argparse
import json

from anatobind.aur.dicom import DicomInputError, inspect_dicom_files


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default error can repeat a private supplied pathname.
        self.exit(2, "invalid_arguments\n")


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("invalid limit")
    try:
        result = inspect_dicom_files(args.input_files[:args.limit])
        result["provided_object_count"] = len(args.input_files)
        result["truncated"] = len(args.input_files) > args.limit
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except DicomInputError as error:
        print(json.dumps({"error": error.code}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
