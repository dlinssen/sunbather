"""
Tests for the sunbather package
"""
import os
import pytest


def f():
    raise SystemExit(1)


def test_import():
    """
    Tests if sunbather can be imported.
    """
    try:
        import sunbather
    except ImportError:
        f()


def test_projectdirs():
    """
    Make sure projectpath exists
    """
    from sunbather import tools
    projectpath = tools.get_sunbather_project_path()
    assert os.path.isdir(
        projectpath
    ), "Please create the projectpath folder on your machine"


def test_planetstxt():
    """
    Make sure the planets.txt file exists
    """
    from sunbather import tools
    projectpath = tools.get_sunbather_project_path()    
    assert os.path.isfile(
        projectpath + "/planets.txt"
    ), "Please make sure the 'planets.txt' file is present in $SUNBATHER_PROJECT_PATH"


def test_seds():
    """
    Make sure the SED we need for this test has been copied to Cloudy
    """
    from sunbather import tools
    assert os.path.isfile(
        tools.get_cloudy_path() + "/data/SED/eps_Eri_binned.spec"
    ), (
        "Please copy /sunbather/stellar_SEDs/eps_Eri_binned.spec "
        "into $CLOUDY_PATH/data/SED/"
    )
