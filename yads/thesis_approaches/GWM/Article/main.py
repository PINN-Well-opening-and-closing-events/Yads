import subprocess as sp

# Generates pressure boundary conditions dataset
sp.call("python article_Pb_dict_gen.py", shell=True)

# Run simulations on the pressure boundary conditions dataset to get a dataset for training
sp.call("python Pimp_generator.py", shell=True)

# fuse data from all generated folders to a single big dataset
sp.call("python fuse_data.py", shell=True)
