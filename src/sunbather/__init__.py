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
    asks for input if not. Also asks to copy the default files to the working dir.

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
                # if quiet, use the current dir
                workingdir = "./"
    if not quiet:
        q = input(f"Copy default files to the working dir ({workingdir})? (y/n) ")
        while q.lower() not in ["y", "n"]:
            q = input("Please enter 'y' or 'n': ")
        if q == "n":
            return

    sunbatherpath = f"{pathlib.Path(__file__).parent.resolve()}"
    for file in os.listdir(f"{sunbatherpath}/data/workingdir"):
        if not os.path.exists(f"{workingdir}/{file}"):
            shutil.copyfile(
                f"{sunbatherpath}/data/workingdir/{file}",
                f"{workingdir}/{file}",
            )
        else:
            if not quiet:
                print("File already exists! Overwrite?")
                q = input("(y/n) ")
                while q.lower() not in ["y", "n"]:
                    q = input("Please enter 'y' or 'n': ")
                if q == "n":
                    continue
            else:
                continue
            shutil.copyfile(
                f"{sunbatherpath}/data/workingdir/{file}",
                f"{workingdir}/{file}",
            )

    return


def firstrun(quiet=False, workingdir=None, cloudy_version="23.01"):
    """
    Runs 'check_cloudy()' and 'make_workingdir()'.
    """
    check_cloudy(quiet=quiet, cloudy_version=cloudy_version)
    make_workingdir(quiet=quiet, workingdir=workingdir)

    print("Sunbather is ready to go!")
