#!/usr/bin/env python
import os
import importlib.util, pathlib

# Import Costum classes
local_path = pathlib.Path(__file__).parent.resolve()
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(local_path)))
pkg_directory = pkg_directory+'/uwmsn-sim'+'/src'+'/Classes'
local_directory = os.path.dirname(os.path.dirname(local_path))+'/src'

spec = importlib.util.spec_from_file_location("module.config", pkg_directory+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


spec = importlib.util.spec_from_file_location("module.bnb", local_directory+'/bnb.py')
bnb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnb)

