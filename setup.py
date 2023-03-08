#refered from: https://youtu.be/tEFkHEKypLI
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'SLiM project and its supporting scripts'
LONG_DESCRIPTION = 'SLiM project calculate the slab like microtearing mode with global effect consideration'

# Setting up
setup(
    name="SLiM",
    version=VERSION,
    author="maxtcurie (Max Curie)",
    author_email="<maxtcurie@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['scipy', 'numpy', 'pandas',\
                    'matplotlib', 'tqdm','sys',\
                    'csv','math','re','os','cmath',\
                    'optparse','tkinter','traceback'],
    keywords=['python', 'plasma', 'physics', \
            'microtearing modes', 'reduced model', \
            'neural network'],
    classifiers=[
        "Development Status :: 1 - Testing",
        "Intended Audience :: Physicist",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
