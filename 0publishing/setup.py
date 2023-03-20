#refered from: https://youtu.be/tEFkHEKypLI

#step1: python setup.py sdist bdist_wheel
#step2: twine upload dist/*

from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'SLiM project and its supporting scripts'
LONG_DESCRIPTION = 'SLiM project calculate the slab like microtearing mode with global effect consideration'

# Setting up
setup(
    name="SLiM_phys",
    version=VERSION,
    author="maxtcurie (Max Curie)",
    author_email="<maxtcurie@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    #install_requires=['scipy', 'numpy', 'pandas',\
    #                'matplotlib', 'tqdm','sys',\
    #                'csv','math','os','cmath',\
    #                'tkinter'],
    keywords=['python', 'plasma', 'physics', \
            'microtearing modes', 'reduced model', \
            'neural network'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    Homepage = "https://github.com/maxtcurie/SLiM/wiki"
)
