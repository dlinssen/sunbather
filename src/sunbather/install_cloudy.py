import os
import pathlib
import urllib.request
import tarfile
import subprocess
import shutil


class GetCloudy:
    """
    Class to download and compile the Cloudy program
    """

    def __init__(self, version="23.01"):
        self.version = version
        self.path = "./"
        major = version.split(".")[0]
        self.url = f"https://data.nublado.org/cloudy_releases/c{major}/"
        self.filename = "c{version}.tar.gz"
        self.sunbatherpath = f"{pathlib.Path(__file__).parent.resolve()}"
        self.cloudypath = f"{self.sunbatherpath}/cloudy/"

    def download(self):
        """
        Creates the cloudy directory and downloads the cloudy version specified.
        """
        if not pathlib.Path(self.cloudypath).is_dir():
            os.mkdir(self.cloudypath)
        else:
            print("Directory already exists! Skipping download.")
            return
        os.chdir(self.cloudypath)
        with urllib.request.urlopen(f"{self.url}{self.filename}") as g:
            with open(self.filename, "b+w") as f:
                f.write(g.read())
        # Go to the v23 download page and download the "c23.01.tar.gz" file
        return

    def extract(self):
        """
        Extracts Cloudy.
        """
        os.chdir(self.cloudypath)
        with tarfile.open(self.filename, "r:gz") as tar:
            tar.extractall(filter="data")

    def compile(self):
        """
        Compiles Cloudy.
        """
        os.chdir(f"{self.cloudypath}/c{self.version}/source/")
        subprocess.Popen(
            [
                "make",
            ]
        ).wait()

    def test(self):
        # Quickly test the Cloudy installation: in the source folder, run ./cloudy.exe, type "test" and hit return twice. It should print "Cloudy exited OK" at the end.
        os.chdir(f"{self.cloudypath}/c{self.version}/source/")
        print(
            'Type "test" and hit return twice. '
            'It should print "Cloudy exited OK" at the end.'
        )
        subprocess.Popen(
            [
                "./cloudy.exe",
            ]
        ).wait()

    def copy_data(self):
        shutil.copy2(
            f"{self.sunbatherpath}/stellar_SEDs/*.spec",
            f"{self.cloudypath}/c{self.version}/data/SED/",
        )
