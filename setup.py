#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from sphinx.setup_command import BuildDoc

import pygeoid

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

name = 'pygeoid'
version = pygeoid.__version__


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


cmdclass = {
    'build_sphinx': BuildDoc,
    'test': PyTest
}

setup(name=name,
      version=version,
      description='Local gravity field modelling with Python',
      long_description=long_description,
      url='https://github.com/ioshchepkov/pygeoid',
      author='Ilya Oshchepkov',
      author_email='ilya.oshchepkov@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 1 - Planning',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Education',
          'Topic :: Scientific/Engineering',
      ],
      keywords=['geodesy', 'gravimetry', 'geoid'],
      packages=find_packages(),
      include_package_data=True,
      tests_require=['pytest'],
      install_requires=['numpy', 'scipy', 'pyproj', 'pint', 'pyshtools',
          'joblib'],
      python_requires='>=3.5',
      cmdclass=cmdclass,
      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'release': ('setup.py', version)}},
      )
