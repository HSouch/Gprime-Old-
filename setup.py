
from distutils.core import setup
from setuptools import find_packages
import galprime

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='galprime',
    packages=find_packages(),
    version=galprime.__version__,
    license='BSD-3-Clause',
    description = 'Simulation suite for efficiency testing of galaxy surface brightness profiles.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Harrison Souchereau",
    author_email='harrison.souchereau@yale.edu',
    url='https://github.com/HSouch/GalPRIME',
    keywords='galaxies surface brightness profiles',
    scripts=[
        "bin/GPRIME"
    ],
    install_requires=[
        'scikit-image',
        'numpy',
        'photutils',
        'astropy',
        'pebble',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
