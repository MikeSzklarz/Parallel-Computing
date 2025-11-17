Here are the parameters I ran for the python code

For the sweep script the environmet requires numpy, matplotlib, and pandas. 
If the movie script is run, then ffmpeg is also required. This was isntalled through conda
The movie script has default parameters and can be run as is, or the parameters can be adjusted in the commandline.

# Sweep script commandline parameters

python3 sweep_mc_slab.py --C 0.5 \
--CC 0.1 \
--H-min 0.5 \
--H-max 25 \
--H-step 0.25 \
--N 10000000 \
--seed 42 \
--trace \
--trace-every 100 \
--make-convergence-plots

python3 sweep_mc_slab.py --C 0.5 \
--CC 0.1 \
--H-min 0.5 \
--H-max 25 \
--H-step 0.25 \
--N 10000 \
--seed 42 \
--trace \
--trace-every 100 \
--make-convergence-plots