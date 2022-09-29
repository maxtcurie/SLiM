# SLiM
The Slab Like Microtearing mode (SLiM) model

Overview

This software provides a rapid assessment of the slab-like microtearing mode using a global linear dispersion model, which takes 50ms to calculate the growth rate and frequency of a given mode on the personal computer. Potentially uses 10^-7 of the computation resources for discharge study. For detail, one can check on the site (under construction): https://www.drmcurie.com/project-page/Research_Projects/SLiM

SLiM EXE can be found from this link: https://drive.google.com/drive/folders/12e1t6liY5JztwOBOLehPoV8GbfORn_j8?usp=sharing



Executable the program: 

1. Plot the modified the safety factor (q) to see if the rational surfaces are intersected with the q profile. 
 GUI:    000GUI_Plot_q_modification.py
 script: 0Plot_q_modification.py

2. Determine the stabilities of the MTM for different mode numbers 
 GUI:    000GUI_SLiM_mode_finder.py
 script: 00SLiM_mode_finder.py

3. Calculate a list of dispersion relations provided by a csv file
 script: 0MTMDispersion_list_Calc.py
 script(CPU accelerated,beta): 0MTMDispersion_list_Calc_parallel.py


GitHub repo:
https://github.com/maxtcurie/SLiM

APS 2021 invited talk about SLiM model:
https://youtu.be/j2MYfGwlBYY

Playlist for tutorial on running the SLiM model:
https://youtube.com/playlist?list=PLgNi5MiqkBWagsB8yRjRncsz1D4oeedQB

How to use GUI:
    mode finder GUI: https://youtu.be/R_-ldYNvmhU
    plot modified safety factor GUI: https://youtu.be/L01xl_e1bpM


CPU accellerated dispersion calculation:
    With    CPU acceleration: 297.5 sec
    Without CPU acceleration: 481.1 sec

Trained neural network dispersion calculation: 0.05sec


Citation 

This software is based on the following articles and presentations, please the cite those articles in the publications uses such software package: 

1. M.T. Curie, J. L. Larakers, D. R. Hatch, A. O. Nelson, A. Diallo, E. Hassan, W. Guttenfelder, M. Halfmoon, M. Kotschenreuther, R. D. Hazeltine, S. M. Mahajan, R. J. Groebner, J. Chen, C. Perez von Thun, L. Frassinetti, S. Saarelma, C. Giroud, M. M. Tennery (2022) "A survey of pedestal magnetic fluctuations using gyrokinetics and a global reduced model for microtearing stability" Physics of Plasmas (Editor's Pick)
https://doi.org/10.1063/5.0084842

2. M. Curie, J.L. Larakers, D.R. Hatch, A. Diallo, E. Hassan, O. Nelson, W. Guttenfelder, M. Halfmoon, M. Kotschenreuthe, S. M. Mahajan, R. J. Groebner (2021)"Reduced predictive models for Micro-tearing modes in the pedestal" APS DPP
https://doi.org/10.13140/RG.2.2.27713.48482

3. M. Curie (2022) "Simulations and reduced models for Micro-tearing modes in the Tokamak pedestal" Ph.D. Dissertation

4. J.L. Larakers,  M. Curie, D. R. Hatch, R. D. Hazeltine, and S. M.Mahajan, 2021) "Global Theory of Microtearing Modes in the Tokamak Pedestal" 
https://doi.org/10.1103/PhysRevLett.126.225001


SLiM_obj.py

self.r_sigma
self.R_ref
self.cs_to_kHz
self.omn
self.omn_nominal
self.cs
self.rho_s
self.Lref
self.x
self.shat
self.shat_nominal
self.eta
self.ky
self.ky_nominal=
self.nu
self.zeff
self.beta
self.q
self.q_nominal
self.ome
self.ome_nominal
self.Doppler
