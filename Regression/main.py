from Test import Test
from Plots import Plot
from Load import Load
import sys


def main():

    if "-t" in sys.argv:
        Test()

    elif "-i" in sys.argv:
        Plot()

    elif "-r" in sys.argv:
        Load()


if __name__ == "__main__":
    main()
