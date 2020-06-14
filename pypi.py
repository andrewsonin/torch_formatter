from os import system

from globals import PYPI_UPLOAD_FILE

system(f'python prepare.py && bash {PYPI_UPLOAD_FILE}')
