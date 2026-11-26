#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import runpy
import sys
from pathlib import Path

script = Path(__file__).with_name('infer.py')
sys.argv = [str(script)] + ['--run' if a == '--train-dir' else a for a in sys.argv[1:]]
runpy.run_path(str(script), run_name='__main__')
