import os
from os.path import abspath, dirname

import mobrob

DATA_DIR = os.path.join(dirname(dirname(dirname(abspath(mobrob.__file__)))), "data")
PROJ_DIR = dirname(abspath(mobrob.__file__))
