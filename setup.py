#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import setuptools
from setuptools import setup
from setuptools.command.test import test as TestCommand
import versioneer


class PyTest(TestCommand):
    description = "Run test suite with pytest"

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        sys.exit(pytest.main(self.test_args))


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "dask[array] >=2023.2.0",
    "dask[dataframe] >=2023.2.0",
    "numpy >=1.18",
    "scipy >=0.19.1",
    "pims >=0.4.1",
    "tifffile >=2018.10.18",
    "pandas >=2.0.0",
]

test_requirements = [
    "flake8 >=3.4.1",
    "pytest >=3.0.5",
    "pytest-flake8 >=0.8.1",
    "pytest-timeout >=1.0.0",
]

cmdclasses = {
    "test": PyTest,
}
cmdclasses.update(versioneer.get_cmdclass())


setup(
    name="dask-image",
    version=versioneer.get_version(),
    description="Distributed image processing",
    long_description=readme + "\n\n" + history,
    author="dask-image contributors",
    url="https://github.com/dask/dask-image",
    cmdclass=cmdclasses,
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=requirements,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="dask-image",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    tests_require=test_requirements
)
