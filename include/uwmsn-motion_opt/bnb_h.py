import os, pathlib, importlib

ws_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve())))
local_directory = ws_directory+"/src/Classes"
external_directory = pkg_directory+"/uwmsn-sim/src/Classes"

spec = importlib.util.spec_from_file_location("module.utils", local_directory+'/utils_opt.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("module.estimator", local_directory+'/estimator.py')
estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator_module)

spec = importlib.util.spec_from_file_location("module.target", local_directory+'/target.py')
target_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target_module)

spec = importlib.util.spec_from_file_location("module.config", external_directory+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location("module.sensor", external_directory+"/sensor.py")
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)