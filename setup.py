from __future__ import print_function
import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from distutils.ccompiler import new_compiler
import os
import sys
import tempfile
def main():
    setup(name="RSR-Superpixel",
          version="0.0.1",
          description="Superpixel implementation of RSR method",
          author="Sam Christian",
		  url='https://github.com/Sam-2727/superpixel_rsr',
		  packages =['superpixel_rsr'],
		  zip_safe=False,
		  classifiers=[
			  "Development Status :: 5 - Production/Stable",
			  "Intended Audience :: Science/Research",
			  "License :: OSI Approved :: MIT License",
			  "Programming Language :: Python :: 3 :: Only",
		  ],
          author_email="samchristian@mit.edu",
		  licence="MIT")

if __name__ == "__main__":
    main()