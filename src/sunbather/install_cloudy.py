import os
import pathlib
import urllib.request
import tarfile
import subprocess


class get_cloudy:
    def __init__(self, version="23.01"):
        self.version = version
        self.path = "./"
        major = version.split(".")[0]
        self.url = f"https://data.nublado.org/cloudy_releases/c{major}/"
        self.filename = "c{version}.tar.gz"
        self.cloudypath = f"{pathlib.Path(__file__).parent.resolve()}/cloudy/"

    def download(self):
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

    def compile(self):
        # Extract it in a location where you want to install Cloudy.
        os.chdir(self.cloudypath)
        tar = tarfile.open(self.filename, "r:gz")
        tar.extractall(filter="data")
        tar.close()

        # cd into the /c23.01/source/ or /c17.02/source/ folder and compile the code by running make.
        os.chdir(f"{self.cloudypath}/c{version}/source/")
        subprocess.Popen(["make",]).wait()

    def test(self):
        # Quickly test the Cloudy installation: in the source folder, run ./cloudy.exe, type "test" and hit return twice. It should print "Cloudy exited OK" at the end.
        os.chdir(f"{self.cloudypath}/c{version}/source/")
        print(
            "Type \"test\" and hit return twice. "
            "It should print \"Cloudy exited OK\" at the end."
        )
        subprocess.Popen(["./cloudy.exe",]).wait()
