from setuptools import setup, find_packages

setup(
    name='daofun',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.6',
        'pandas==1.3.5',
        'matplotlib==3.5.3',
        'astropy==4.3.1',
        'aplpy==2.1.0',
        'IPython==7.34.0',
        'PySimpleGUI==4.60.5',
        'spectral-cube',
        'tqdm==4.66.1',
        'specutils==1.8.1'
    ],
    entry_points={
        'console_scripts': [
            'daofun = daofun.daofun:run_daofun',
            'daocube = daocube.daocube:run_daocube',  # Puede variar dependiendo de cómo definas tu función principal
            'specfun = specfun.specfun:run_specfun',
        ]
    },
    # Otros metadatos como autor, descripción, etc.
    author='Carlos Quezada',
    description="""
DAOFUN is an interactive adaptation of the DAOPHOT 
astronomical image processing software. It provides 
a user-friendly interface through button-based interactions, 
enabling astronomers to perform various tasks seamlessly. 
The software operates by executing DAOPHOT commands in the 
background, facilitating astronomical data analysis, such 
as finding, photometry, and refining point spread functions (PSFs) 
in images. DAOFUN simplifies the utilization of DAOPHOT's capabilities 
by integrating them into an accessible and intuitive graphical user interface.


Credits: by Carlos Quezada
            inspired in the work of Alvaro Valenzuela
            thanks to DAOPHOT by Peter Stetson""",
    license='MIT',
    keywords=['daophot', 'astronomical', 'python'],
    url='https://github.com/ciquezada/daofun'
)
