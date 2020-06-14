from inspect import cleandoc
from os import chdir
from os.path import isdir
from shutil import rmtree
from typing import Final

from globals import *

chdir(PROJECT_DIR)

for setup_dir in filter(isdir, SETUP_DIRS):
    rmtree(setup_dir)

conda_yml_content: Final = cleandoc(
    f"""
    package:
      name: chained
      version: {VERSION}

    build:
      number: 1

    requirements:
      build:
        - python{PYTHON_VERSION}
        - setuptools
      run:
        - python{PYTHON_VERSION}
    about:
      summary: {SLOGAN}
    """
)
conda_sh_content = cleandoc(
    """
    #!/usr/bin/env bash
    python setup.py install
    """
)
conda_bat_content = cleandoc(
    """
    "%PYTHON%" setup.py install
    if errorlevel 1 exit 1
    """
)

with open(CONDA_YML_FILE, 'w') as out:
    out.write(conda_yml_content)
with open(CONDA_SH_FILE, 'w') as out:
    out.write(conda_sh_content)
with open(CONDA_BAT_FILE, 'w') as out:
    out.write(conda_bat_content)

with open(PYPI_PWSD_FILE, 'r') as in_file:
    PYPI_PSWD = in_file.read().strip()

pypi_upload_script_content = cleandoc(
    f"""
    #!/usr/bin/env bash
    set -e
    python setup.py sdist
    python setup.py bdist_wheel
    twine upload -r pypi -u {PYPI_USERNAME} -p {PYPI_PSWD} dist/*
    """
)
with open(PYPI_UPLOAD_FILE, 'w') as out:
    out.write(pypi_upload_script_content)
