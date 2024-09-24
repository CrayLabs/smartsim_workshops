from collections import defaultdict

import os

from smartsim import Experiment

def get_db_settings():
    db_settings = defaultdict(lambda: {"interface":"lo0"})
    db_settings["frontier"] = {"interface": ["hsn0","hsn1","hsn2","hsn3"]}
    db_settings["perlmutter"] = {"interface": ["hsn0","hsn1","hsn2","hsn3"]}
    system_name = os.environ.get("LMOD_SYSTEM_NAME", None)
    return db_settings[system_name]

def main():

    exp = Experiment("online-training")

    # Create the objects for the mock simulation
    rs_sim = exp.create_run_settings("build/mock_cpp_simulation")
    rs_sim.set_tasks(1)
    model = exp.create_model("mock_simulation", rs_sim)
    model.attach_generator_files(to_symlink=["data"])

    # Create the objects for the sampler
    rs_sampler = exp.create_run_settings("python", exe_args="sampler.py")
    rs_sampler.set_tasks(1)
    sampler = exp.create_model("sampler", rs_sampler)
    sampler.attach_generator_files(to_symlink=["sampler.py"])

    # Create the objects for the trainer
    rs_trainer = exp.create_run_settings("python", exe_args="trainer.py")
    rs_trainer.set_tasks(1)
    trainer = exp.create_model("trainer", rs_trainer)
    trainer.attach_generator_files(to_symlink=["trainer.py"])

    db = exp.create_database(**get_db_settings())
    exp.start(db)

    try:
        exp.generate(model, sampler, trainer, overwrite=True)
        exp.start(model, sampler, trainer, block=True)
    finally:
        exp.stop(db)

if __name__ == "__main__":
    main()