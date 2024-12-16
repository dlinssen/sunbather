"""
Initialize sunbather
"""
import os
import pathlib
import shutil

import sunbather.tools
from sunbather.install_cloudy import GetCloudy


def check_cloudy(quiet=False, cloudy_version="23.01"):
    """
    Checks if Cloudy executable exists, and if not, prompts to download and build it.
    :quiet: bool, if True, does not ask for input
    :cloudy_version: str, Cloudy version (default: "23.01", environment variable
    CLOUDY_VERSION overrides this)
    """
    try:
        cloudy_version = os.environ["CLOUDY_VERSION"]
    except KeyError:
        pass
    sunbatherpath = os.path.dirname(
        os.path.abspath(__file__)
    )  # the absolute path where this code lives
    try:
        # the path where Cloudy is installed
        cloudypath = os.environ["cloudy_path"]
    except KeyError:
        cloudypath = f"{sunbatherpath}/cloudy/c{cloudy_version}"
    if not os.path.exists(f"{cloudypath}/source/cloudy.exe"):
        if not quiet:
            q = input(
                f"Cloudy not found and CLOUDY_PATH is not set. Do you want to install "
                f"Cloudy {cloudy_version} now in the sunbather path? (y/n) "
            )
            while q.lower() not in ["y", "n"]:
                q = input("Please enter 'y' or 'n'")
            if q == "n":
                raise KeyError(
                    "Cloudy not found, and the environment variable 'CLOUDY_PATH' is "
                    "not set. Please set this variable in your .bashrc/.zshrc file "
                    "to the path where the Cloudy installation is located. "
                    "Do not point it to the /source/ subfolder, but to the main folder."
                )
        installer = GetCloudy(version=cloudy_version)
        installer.download()
        installer.extract()
        installer.compile()
        installer.test()
        installer.copy_data()


def make_workingdir(workingdir=None, quiet=False):
    """
    Checks if the SUNBATHER_PROJECT_PATH environment variable has been set and
    asks for input if not.
    If quiet is True and the working dir is not set, the current dir is used.
    Copies the planets.txt file to the working dir if it does not exist..

    :workingdir: str, path to the working dir. If None, checks the
    SUNBATHER_PROJECT_PATH environment variable, and asks for input if this is
    not set. (default: None)
    :quiet: bool, if True, does not ask for input (default: False)
    """
    if workingdir is None:
        try:
            workingdir = os.environ["SUNBATHER_PROJECT_PATH"]
        except KeyError:
            if not quiet:
                workingdir = input("Enter the working dir for Sunbather: ")
            else:
                # if quiet, use the current dir (absolute path)
                workingdir = os.path.abspath(".")
            os.environ["SUNBATHER_PROJECT_PATH"] = workingdir
            print(
                f"Environment variable SUNBATHER_PROJECT_PATH set to {workingdir}"
            )
    if not os.path.exists(f"{workingdir}/planets.txt"):
        sunbatherpath = f"{pathlib.Path(__file__).parent.resolve()}"
        shutil.copyfile(
            f"{sunbatherpath}/data/workingdir/planets.txt",
            f"{workingdir}/planets.txt",
        )


def firstrun(quiet=False, workingdir=None, cloudy_version="23.01"):
    """
    Runs 'check_cloudy()' and 'make_workingdir()'.
    """
    check_cloudy(quiet=quiet, cloudy_version=cloudy_version)
    make_workingdir(quiet=quiet, workingdir=workingdir)

    print("Sunbather is ready to go!")
