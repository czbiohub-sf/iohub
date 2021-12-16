from setuptools import setup, find_packages
from pathlib import Path
import os
import platform

packages = [package for package in find_packages()]

if platform.machine() == 'ppc64le':
        requirements = []
else:
        requirements = ['numpy>=1.17.4',
                        'pywavelets>=1.1.1',
                        'ipywidgets>=7.5.1',
                        'tifffile>=2020.11.26',
                        'zarr>=2.6.1',
                        'natsort>=7.1.1',
                        'opencv-python>=3.4.1',
                        'scipy>=1.3.0',
                        'matplotlib>=3.1.1']

dir_ = Path(__file__).parent
with open(os.path.join(dir_, 'README.md')) as file:
        long_description = file.read()

setup(  name             = 'waveorder',
        version          = '1.0.0-beta',
        description      = 'wave optical simulations and deconvolution of optical properties',
        author           = 'Li-Hao Yeh',
        author_email     = 'lihao.yeh@czbiohub.org',
        url              = 'https://github.com/mehta-lab/waveorder/',
        license          = 'BSD License (Chan Zuckerberg Biohub Software License)',
        license_file     = 'LICENSE.txt',
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        packages         = packages,
        include_package_data = True,
        python_requies   = '==3.7',
        install_requires = requirements,
        classifiers = [
                'Development Status :: 4 - Beta',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Python :: 3 :: Only',
                'Programming Language :: Python :: 3.7',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Image Processing',
                'Topic :: Scientific/Engineering :: Visualization',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Topic :: Scientific/Engineering :: Bio-Informatics',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS',
        ]
     )
