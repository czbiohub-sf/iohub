# waveorder
This is a package developed to simulate and deconvolve polarization measurements under various partially coherent illumination patterns considering diffraction effects. This generalized wave simulator/reconstructor enable us to develop a new label-free imaging method, __uniaxial permittivity tensor imaging (uPTI)__, that reconstructs density (phase) and 3D anisotropy (principal retardance, 3D orientation of the optic axis, and optic sign) of specimens from asymmetrically illuminated and polarization-resolved images. The acquisition, calibration, background correction, and reconstruction algorithms are described in the following preprint:

```Yeh, L.-H., Ivanov, I. E., Chhun, B. B., Guo, S.-M., Hashemi, E., Byrum, J. R., Pérez-Bermejo, J. A., Wang, H., Yu, Y., Kazansky, P. G., Conklin, B. R., Han, M. H., Mehta, S. B. (2019). uPTI: uniaxial permittivity tensor imaging of intrinsic density and anisotropy, BioRxiv XXXXXX.<br/>```

As an illustration, following figure shows the simulated and reconstructed data of uPTI using `waveorder`: 

![Data_flow](Fig_Readme.png)

This package also enables simulations and reconstructions of other label-free imaging (phase, polarization) modality under partially coherent illumination such as:

1. 2D/3D phase reconstruction with a single brightfield defocused stack (Transport of intensity, TIE)
    
2. 2D/3D phase reconstruction with intensities of asymetric illumination (differential phase contrast, DPC)
       
3. 2D/3D joint phase and polarization (2D orientation) reconstruction with brightfield-illuminated polarization-sensitive intensities (QLIPP)

## Installation

### Create a new conda environment
Install conda package management system by installing anaconda or miniconda ([link](https://conda.io/)). 

1) Creating a conda environment dedicated to `waveorder` will avoid version conflicts among packages required by `waveorder` and packages required by other python software.
>```buildoutcfg
>conda create -n <your-environment-name> python=3.7
>conda activate <your-environment-name> (or source activate <your-environment-name>)
>```

2) Then, install jupyter notebook with
>```buildoutcfg
>conda install jupyter
>```
    
### Install `waveorder` and required packages
Install the git version control system git : [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

1) Use git to clone this repository to your current directory:
>```buildoutcfg
>git clone https://github.com/mehta-lab/waveorder.git
>```

2) Then, you can install dependencies via pip (python index package) <br>
  
>    If you are running waveorder on your own machine, <br>
>
>    a) navigate to the cloned repository:
>
>    ```buildoutcfg
>    cd waveorder
>    ```
>    <br>
>    b) install python library dependencies:
>
>    ```buildoutcfg
>    pip install -r requirements.txt
>    ```
>    <br>

3) Create a symbolic library link with setup.py:
>
>```buildoutcfg
>python setup.py develop
>```

*`waveorder` supports NVIDIA GPU computation through cupy package, please follow [here](https://github.com/cupy/cupy) for installation (check cupy is properly installed by ```import cupy```). To enable gpu processing, set ```use_gpu=True``` when initializing the simulator/reconstructor class.*


## Usage and example

In the following, we demonstrate how to run `waveorder` for simulation and reconstruction. <br>

1) In the terminal, switch to the environment with waveorder installed 
>  ```buildoutcfg
>  conda activate <your-environment-name>
>  ```

2) Navigate to the repository folder:
>  ```buildoutcfg
>  cd waveorder/example
>  ```

3) Open jupyter notebook or lab to run the simulation/reconstruction notebook in the folder:
>  ```buildoutcfg
>  jupyter notebook
>  ```
    
Currently, the example notebooks contain simulations for 2D QLIPP, 3D PODT, and 2D/3D uPTI. There is also a notebook demonstrating the reconstruction of the experimental data of 3D uPTI (data will be uploaded soon).
    
## License
Chan Zuckerberg Biohub Software License

This software license is the 2-clause BSD license plus clause a third clause
that prohibits redistribution and use for commercial purposes without further
permission.

Copyright © 2019. Chan Zuckerberg Biohub.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3.	Redistributions and use for commercial purposes are not permitted without
the Chan Zuckerberg Biohub's written permission. For purposes of this license,
commercial purposes are the incorporation of the Chan Zuckerberg Biohub's
software into anything for which you will charge fees or other compensation or
use of the software to perform a commercial service for a third party.
Contact ip@czbiohub.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 