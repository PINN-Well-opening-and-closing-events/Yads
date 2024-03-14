import subprocess as sp 

# run data generation script 
sp.call('python data_generation.py', shell=True)

# Fuse all data files and clean folder 
sp.call('python fuse_data_and_clean.py', shell=True)

