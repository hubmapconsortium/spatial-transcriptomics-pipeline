#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import starfish


def main(data_dir: Path):
    print("hello world")
    pass


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("data_dir", type=Path)
    args = p.parse_args()

    main(args.data_dir)
