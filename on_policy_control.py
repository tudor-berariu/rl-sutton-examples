from argparse import Namespace
from liftoff.config import read_config


def run(args: Namespace):
    pass


def main():
    args = read_config()
    run(args)


if __name__ == "__main__":
    main()
