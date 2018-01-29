"""
Setup script for causalbenchmark a causal inference benchmarking framework.


(C) IBM Corp, 2018, All rights reserved
Created on Jan 18, 2018

@author: EHUD KARAVANI
"""
from __future__ import print_function
from setuptools import setup, find_packages
import os.path
import sys
from causalbenchmark import __version__ as causalbenchmark_version


GIT_URL = "https://github.com/IBM-HRL-MLHLS/IBM-Causality-Benchmarking-Framework"
MANIFEST_FILE_PATH = "MANIFEST.in"
LICENSE_FILE_PATH = "License.txt"
README_FILE_PATH = "README.md"
REQUIREMENTS_FILE_PATH = "requirements.txt"


# ########################## #
# Create setup related files #
# ########################## #

if os.path.exists(REQUIREMENTS_FILE_PATH):
    os.remove(REQUIREMENTS_FILE_PATH)
with open(REQUIREMENTS_FILE_PATH, "w") as fh:
    fh.write("pandas>=0.20.3,<1.0\n")
    fh.write("numpy>=1.13.1,<2.0\n")
    fh.write("future>=0.16.0,<1.0\n")

if os.path.exists(MANIFEST_FILE_PATH):
    os.remove(MANIFEST_FILE_PATH)
with open(MANIFEST_FILE_PATH, "a") as fh:
    fh.write("include {} \n".format(README_FILE_PATH))
    fh.write("include {} \n".format(LICENSE_FILE_PATH))
    fh.write("include {} \n".format(REQUIREMENTS_FILE_PATH))
    fh.write("include {} \n".format("setup.py"))
    fh.write("include {} \n".format("setup.cfg"))

# Declare files for the setup:
package_data = {"": [],
                "causalbenchmark": ['*.txt', '*.md', '*.yml']}               # Any text files within package.
data_files = [(".", [LICENSE_FILE_PATH, REQUIREMENTS_FILE_PATH, MANIFEST_FILE_PATH])]


# ########################################## #
# Minimalistic command-line argument parsing #
# ########################################## #

# Whether to exclude benchmark's data files from setup:
INCLUDE_DATA_CMD_FLAG = "--include-data"
if INCLUDE_DATA_CMD_FLAG in sys.argv:
    print("\t Including data files from setup")
    data_files.append(("data", ["data/*.csv"]))
    package_data[""].append("data/*.csv")
    with open(MANIFEST_FILE_PATH, "a") as fh:
        fh.write("graft data \n")                           # To add files outside the package tree
    sys.argv.remove(INCLUDE_DATA_CMD_FLAG)
    print("Warning: This is not fully supported and may cause later installation to fail.", file=sys.stderr)

# Whether to include benchmark's unittests (it requires downloading some dummy data)
INCLUDE_TESTS_CMD_FLAG = "--include-tests"
if INCLUDE_TESTS_CMD_FLAG in sys.argv:
    print("\t Including tests from setup")
    data_files.append(("tests", ["tests/*.csv", "tests/*.py"]))
    package_data[""].append("tests/*.csv")
    package_data[""].append("tests/*.py")
    with open(MANIFEST_FILE_PATH, "a") as fh:
        fh.write("graft tests \n")                          # To add files outside the package tree
        # fh.write("recursive-include tests/.*csv tests/*.py")
    sys.argv.remove(INCLUDE_TESTS_CMD_FLAG)
    print("Warning: This is not fully supported and may cause later installation to fail.", file=sys.stderr)


# ######## #
# Do setup #
# ######## #

setup(name='causalbenchmark',
      version=causalbenchmark_version,
      license=open(os.path.join(os.path.dirname(__file__), LICENSE_FILE_PATH)).read(),
      packages=find_packages(),
      description="causalbenchmark is a framework for evaluating methods that infer causal effect from observational "
                  "data",
      long_description=open(os.path.join(os.path.dirname(__file__), README_FILE_PATH)).read(),

      author="Ehud Karavani",
      author_email="ehudk@ibm.com",
      keywords="causal-inference causality benchmarking evaluations effect-size-estimation",
      url=GIT_URL,
      project_urls={"Documentation": GIT_URL + "/wiki",
                    "Source Code": GIT_URL},

      install_requires=["pandas>=0.20.3,<1.0",
                        "numpy>=1.13.1,<2.0",
                        "future>=0.16.0"],

      package_data=package_data,
      data_files=data_files,
      include_package_data=True

      # entry_points={'console_scripts': ['causalbenchmark = causalbenchmark.evaluate:main']}
      )
