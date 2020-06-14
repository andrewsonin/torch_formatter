from typing import Final

from setuptools import setup, find_packages  # type: ignore

from globals import *

with open(README_FILE, 'r') as readme:
    long_desc: Final = readme.read()

setup(
    name=PROJECT_NAME,
    version=VERSION,
    url=PROJECT_URL,
    license=LICENCE,
    author=AUTHOR,
    author_email=EMAIL,
    description=SLOGAN,
    packages=find_packages(exclude=('test',)),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Operating System :: OS Independent',
        'Typing :: Typed',
        'Development Status :: 2 - Pre-Alpha',
        f'License :: OSI Approved :: {LICENCE}',
    ],
    long_description=long_desc,
    long_description_content_type='text/markdown',
    zip_safe=True,
    python_requires=PYTHON_VERSION
)
