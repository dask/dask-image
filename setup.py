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

requirements = [
    "dask",
    "numpy",
    "pims",
]

test_requirements = [
    "pytest",
    "scikit-image",
]

cmdclasses = {
    "test": PyTest,
}
cmdclasses.update(versioneer.get_cmdclass())

setup(
    name="dask-imread",
    version=versioneer.get_version(),
    description="A library for loading image data into Dask",
    long_description=readme,
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    url="https://github.com/dask-image/dask-imread",
    cmdclass=cmdclasses,
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=requirements,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="dask-imread",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    tests_require=test_requirements
)
