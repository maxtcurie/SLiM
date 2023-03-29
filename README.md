# SLiM
The Slab Like Microtearing mode (SLiM) model

## Overview

This software provides a rapid assessment of the slab-like microtearing mode using a global linear dispersion model, which takes 50ms to calculate the growth rate and frequency of a given mode on the personal computer. Potentially uses 10^-7 of the computation resources for discharge study. For detail, one can check on the site (under construction): https://www.drmcurie.com/project-page/Research_Projects/SLiM

The package can be download via pip: 
`pip install slim_phys`

GitHub repo:
https://github.com/maxtcurie/SLiM

for more detailed information, please check wikipage: 
[https://github.com/maxtcurie/SLiM/wiki](https://github.com/maxtcurie/SLiM/wiki)

## Executable the program: 

1. Plot the modified the safety factor (q) to see if the rational surfaces are intersected with the q profile. 
 *GUI:    `GUI/Plot_q_modification.py`   (under the folder GUI)
 *script: `0Plot_q_modification.py`

2. Determine the stabilities of the MTM for different mode numbers 
 *GUI:    `GUI/mode_finder.py`           (under the folder GUI)
 *script: `00SLiM_mode_finder.py`

3. Calculate dispersion 
 *script: `Templet/Dispersion.py`        (under the folder Templet)



## Tutorial

### APS 2021 invited talk about SLiM model:
https://youtu.be/j2MYfGwlBYY

### Playlist for tutorial on running the SLiM model:
https://youtube.com/playlist?list=PLgNi5MiqkBWagsB8yRjRncsz1D4oeedQB

### How to use GUI:
#### mode finder GUI: https://youtu.be/R_-ldYNvmhU
    
#### plot modified safety factor GUI: https://youtu.be/L01xl_e1bpM


## Citation 

This software is based on the following articles and presentations, please the cite those articles in the publications uses such software package: 

1. M.T. Curie, J. L. Larakers, D. R. Hatch, A. O. Nelson, A. Diallo, E. Hassan, W. Guttenfelder, M. Halfmoon, M. Kotschenreuther, R. D. Hazeltine, S. M. Mahajan, R. J. Groebner, J. Chen, C. Perez von Thun, L. Frassinetti, S. Saarelma, C. Giroud, M. M. Tennery (2022) "A survey of pedestal magnetic fluctuations using gyrokinetics and a global reduced model for microtearing stability" Physics of Plasmas (Editor's Pick)
https://doi.org/10.1063/5.0084842

2. J.L. Larakers,  M. Curie, D. R. Hatch, R. D. Hazeltine, and S. M.Mahajan, 2021) "Global Theory of Microtearing Modes in the Tokamak Pedestal" 
https://doi.org/10.1103/PhysRevLett.126.225001

3. M. Curie, J.L. Larakers, D.R. Hatch, A. Diallo, E. Hassan, O. Nelson, W. Guttenfelder, M. Halfmoon, M. Kotschenreuthe, S. M. Mahajan, R. J. Groebner (2021)"Reduced predictive models for Micro-tearing modes in the pedestal" APS DPP
https://doi.org/10.13140/RG.2.2.27713.48482

4. M. Curie (2022) "Simulations and reduced models for Micro-tearing modes in the Tokamak pedestal" Ph.D. Dissertation
https://doi.org/10.13140/RG.2.2.24468.37769
