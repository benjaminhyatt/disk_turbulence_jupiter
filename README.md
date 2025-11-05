# disk_turbulence_jupiter

This repository is for storing scripts being used to run (jupiter-run), process (jupiter-process), and plot (jupiter-plot) data from simulations in the Dedalus code. 

Our current simulation setup uses a Fourier-Zernike basis for the full unit disk in polar coordinates to solve forced-dissipative incompressible momentum equations with the Coriolis force represented under the gamma-plane approximation. We have so far developed processing scripts to: 
- track fundamental scalar quantities such as mean kinetic energy and enstrophy, as well as azimuthally-averaged profiles and full snapshots of vorticity/potential vorticity 
- map simulation outputs from the Fourier-Zernike basis to a Fourier-Bessel basis for creating 2d spectra of kinetic energy and enstrophy

We are currently working on implementing: 
- processing algorithms for tracking (anti)cyclone centers and obtaining empirical distributions of their displacement from the axis of rotation
- simulations with nonzero initial conditions to investigate the effects of nontrivial potential vorticity homogenization
