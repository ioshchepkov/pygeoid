#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages

import versioneer

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

name = 'pygeoid'
version = versioneer.get_version()

#cmdclass = {
#    'build_sphinx': BuildDoc,
cmdclass = versioneer.get_cmdclass()
#}

setup(name=name,
      version=version,
      cmdclass=cmdclass,
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
      setup_requires=['pytest-runner'],
      install_requires=['numpy', 'scipy', 'pyproj', 'pint', 'pyshtools',
          'joblib'],
      python_requires='>=3.5')
