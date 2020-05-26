#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

base_directory = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(base_directory, 'requirements', 'default.txt')
with open(filename) as f:
    contents = f.read()
requirements = contents.split('\n')

filename = os.path.join(base_directory, 'requirements', 'test.txt')
with open(filename) as f:
    contents = f.read()
test_requirements = contents.split('\n')

cmdclasses = {
    "test": PyTest,
}
cmdclasses.update(versioneer.get_cmdclass())


setup(
    name="dask-image",
    version=versioneer.get_version(),
    description="Distributed image processing",
    long_description=readme + "\n\n" + history,
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    url="https://github.com/dask/dask-image",
    cmdclass=cmdclasses,
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    python_requires='>=3.5',
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    tests_require=test_requirements
)
