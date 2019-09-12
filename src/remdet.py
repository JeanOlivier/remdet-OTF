#!/bin/python
# -*- coding: utf-8 -*-

import os as _os
import platform as _platform

# Setting up the proper libraries and paths, mainly for Windows support
_libpath = _os.path.abspath(_os.path.dirname(__file__))
_plat_info = dict(plat=_platform.system())
if _plat_info['plat'] == 'Windows':
    _plat_info['lib'] = _os.path.join(_libpath, 'remdet_wrapper.pyd')
    _plat_info['com'] = 'make remdet_wrapper.pyd'
    # Adding cygwin libs path for windows
    _libspath = 'C:\\cygwin64\\usr\\x86_64-w64-mingw32\\sys-root\\mingw\\bin'
    if _libspath not in _os.environ['PATH']:
        _os.environ['PATH'] = _libspath+_os.path.pathsep+_os.environ['PATH']   
else:
    _plat_info['lib'] = _os.path.join(_libpath, 'remdet_wrapper.so')
    _plat_info['com'] = 'make remdet_wrapper.so'

if not _os.path.isfile(_plat_info['lib']):
    raise IOError("{lib} is missing. To compile on {plat}:\n{com}\n".format(**_plat_info))

from remdet_wrapper import *
