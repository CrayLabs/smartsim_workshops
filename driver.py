from collections import defaultdict

import os

from smartsim import Experiment

db_settings = defaultdict(lambda: {"interface":"lo0"})
db_settings["frontier"] = {"interface": ["hsn0","hsn1","hsn2","hsn3"]}
db_settings["perlmutter"] = {"interface": ["hsn0","hsn1","hsn2","hsn3"]}

system_name = os.environ.get("LMOD_SYSTEM_NAME", None)


exp = Experiment("online-training")

rs = exp.create_run_settings("build/mock_cpp_simulation")
rs.set_tasks(1)

model = exp.create_model("mock_simulation", rs)
model.attach_generator_files(to_symlink=["data"])

db = exp.create_database(**db_settings[system_name])
exp.start(db)

try:
    exp.generate(model, overwrite=True)
    exp.start(model, block=True)
finally:
    exp.stop(db)