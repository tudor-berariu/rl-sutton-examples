import numpy as np


def main():
    tilings_no = 7  # different overlapping tiles
    tiles_no = 5  # how many tiles in each tiling
    inputs_no = 4  # number of input dimensions

    x = np.random.randint(0, 24, size=(inputs_no,))

    offset = np.random.randint(0, 4, size=(tilings_no, inputs_no))

    widths = np.random.randint(4, 7, size=(tilings_no, inputs_no))

    offsetted = (x + offset)

    codes = offsetted // widths

    print("x = ", x)
    print("offset = ", offset)
    print("widths = ", widths)
    print("offsetted = ", offsetted)
    print("codes = ", codes)


if __name__ == "__main__":
    main()
