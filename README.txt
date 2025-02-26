Code to reproduce all results figures from the paper
"On the validity of electric brain signal predictions based on population firing rates"
by Torbjørn V. Ness, Tom Tetzlaff, Gaute T. Einevoll, and David Dahmen

Installation has not been thoroughly tested on different systems, but on Ubuntu
with virtual environments this worked for me:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System dependency: NEST Simulator
# Can for example be installed via:
sudo add-apt-repository ppa:nest-simulator/nest
sudo apt-get update
sudo apt-get install nest

Some simulations are time-consuming, and MPI and patience is recommended
mpirun -n 6 --use-hwthread-cpus python3 make_all_figures.py