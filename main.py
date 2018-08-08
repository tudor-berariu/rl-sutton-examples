from argparse import Namespace
import importlib
from liftoff.config import read_config


def main():
    args: Namespace = read_config()
    module = importlib.import_module(args.script)
    module.run(args)


if __name__ == "__main__":
    main()
