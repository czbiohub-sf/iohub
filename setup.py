from setuptools import setup, find_packages

setup(  name             = 'waverder',
        version          = '0.0.1',
        description      = 'functions for reconstructing and visualizing phase and birefrigence',
        author           = 'Li-Hao Yeh',
        author_email     = 'lihao.yeh@czbiohub.org',
        license          = 'BSD License',
        packages         = find_packages(),
        install_requires = ['numpy', 'matplotlib', 'scipy', 'ipywidgets']
     )