from packaging import version
from platform import python_version

import importlib
import sys

OK = '\x1b[42m[ OK ]\x1b[0m'
FAIL = '\x1b[41m[FAIL]\x1b[0m'


def run_checks():
    """
        Check that the packages we need are installed
        and the Python version is high enough.
    """
    # check the python version
    print(f"Using Python in {sys.prefix}:")

    if version.parse(python_version()) >= version.parse('3.6'):
        print((f"{OK} Python is version {sys.version}\n"))
    else:
        print(
            f"{FAIL} Python version 3.6+ is required, \
                but {sys.version} is installed.\n")

    # read in the requirements
    with open('../requirements.txt', 'r') as file:
        requirements = []

        for pkg in file.read().splitlines():
            requirements.append(pkg)

    # check the requirements
    for pkg in requirements:
        try:
            mod = importlib.import_module(pkg)
            print(f"{OK} {mod.__name__}")
        except ImportError:
            print(f"{FAIL} {pkg} not installed.")


if __name__ == '__main__':
    run_checks()
