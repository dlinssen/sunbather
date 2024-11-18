"""
Initialize sunbather
"""
import os
import pathlib
import shutil

import sunbather.tools

def check_cloudy():
    """
    Checks if Cloudy executable exists, and if not, prompts to download and build it.
    """
    try:
        CLOUDYVERSION = os.environ["CLOUDY_VERSION"]
    except KeyError:
        CLOUDYVERSION = "23.01"
    SUNBATHERPATH = os.path.dirname(
        os.path.abspath(__file__)
    )  # the absolute path where this code lives
    try:
        # the path where Cloudy is installed
        CLOUDYPATH = os.environ["CLOUDY_PATH"]
    except KeyError as exc:
        CLOUDYPATH = f"{SUNBATHERPATH}/cloudy/c{CLOUDYVERSION}"
    if not os.path.exists(f"{CLOUDYPATH}/source/cloudy.exe"):
        q = input(
            f"Cloudy not found and CLOUDY_PATH is not set. "
            f"Do you want to install Cloudy {CLOUDYVERSION} now in the Sunbather path? "
            f"(y/n) "
        )
        while q.lower() not in ["y", "n"]:
            q = input("Please enter 'y' or 'n'")
        if q == "n":
            raise KeyError(
                "Cloudy not found, and the environment variable 'CLOUDY_PATH' is not set. "
                "Please set this variable in your .bashrc/.zshrc file "
                "to the path where the Cloudy installation is located. "
                "Do not point it to the /source/ subfolder, but to the main folder."
            ) from exc
        from sunbather.install_cloudy import GetCloudy
        INSTALLER = GetCloudy(version=CLOUDYVERSION)
        INSTALLER.download()
        INSTALLER.extract()
        INSTALLER.compile()
        INSTALLER.test()
        INSTALLER.copy_data()


def make_workingdir():
    """
    Checks if the SUNBATHER_PROJECT_PATH environment variable has been set and
    asks for input if not. Also asks to copy the default files to the working dir.
    """
    try:
        workingdir = os.environ["SUNBATHER_PROJECT_PATH"]
    except KeyError:
        workingdir = input("Enter the working dir for Sunbather: ")
    q = input(f"Copy default files to the working dir ({workingdir})? (y/n) ")
    while q.lower() not in ["y", "n"]:
        q = input("Please enter 'y' or 'n': ")
    if q == "n":
        return

    sunbatherpath = f"{pathlib.Path(__file__).parent.resolve()}"
    shutil.copytree(
        sunbatherpath + "/data/workingdir",
        workingdir,
    )


def firstrun():
    """
    Runs 'check_cloudy()' and 'make_workingdir()'.
    """
    check_cloudy()
    make_workingdir()
