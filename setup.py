"""
Setup file for package `azimuthal_profile`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'azimuthal_profile'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='Azimuthal profile functions from Birnstiel+2013',
        version='0.0.1',
        long_description=(HERE / "README.md").read_text(),
        long_description_content_type='text/markdown',
        url='https://github.com/birnstiel/azimuthal_profile',
        author='Til Birnstiel',
        author_email='til.birnstiel@lmu.de',
        license='GPLv3',
        packages=[PACKAGENAME],
        package_dir={PACKAGENAME: PACKAGENAME},
        install_requires=[
            'numpy',
            'matplotlib',
            'astropy'],
        zip_safe=False,
        )
