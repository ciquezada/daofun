# DAOFUN - Astronomical Image Processing Project

## Description
DAOFUN is a graphical interface for DAOPHOT, a tool for astronomical image processing. This interface provides an intuitive way to interact with DAOPHOT, allowing astronomers to perform tasks such as object detection, photometry, and refining point spread functions (PSFs) in images.

## Key Features
- Intuitive button-based graphical interface for executing DAOPHOT commands.
- Facilitates common astronomical data analysis tasks.
- Performs functions such as object detection, photometry, and PSF refinement.

## Credits
- **Developer:** Carlos Quezada
- Inspired by the work of Alvaro Valenzuela
- Utilizes DAOPHOT software developed by Peter Stetson

## Usage
The program is executed via the main script `daofun.py`. It provides an intuitive graphical interface for performing various astronomical image processing tasks.

### Dependencies
The project relies on the following libraries and tools:
- `pandas`
- `matplotlib`
- `astropy`
- `aplpy`
- `IPython`
- `PySimpleGUI`

The following libraries were developed/adapted during the project:
- `daophot_wraps`
- `daofun_gui_selection`
- `daofun_backend`
- `fits_handler`

### Installation
1. Clone this repository.
2. Ensure all dependencies are installed, you can run `setup.py`.
3. Run `daofun.py` to launch the graphical interface.

## Usage Instructions
1. Run the program with `python daofun.py`.
2. Select a `.fits` file to start processing astronomical images.
3. Use the provided buttons to perform actions such as object detection, photometry, PSF refinement, and more.

## Contributions
Contributions are welcome! If you wish to contribute, follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/new-feature`.
3. Make your changes and commit: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Open a pull request on GitHub.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.


# (SPANISH) DAOFUN - Proyecto de Procesamiento de Imágenes Astronómicas

## Descripción
DAOFUN es una interfaz gráfica para DAOPHOT, una herramienta de procesamiento de imágenes astronómicas. Esta interfaz proporciona una forma intuitiva de interactuar con DAOPHOT, permitiendo a los astrónomos realizar tareas como encontrar, realizar fotometría y refinar funciones de dispersión de puntos (PSFs) en imágenes.

## Características principales
- Interfaz gráfica intuitiva basada en botones para ejecutar comandos DAOPHOT.
- Facilita tareas comunes de análisis de datos astronómicos.
- Realiza funciones como encontrar objetos, realizar fotometría y refinar PSFs.

## Créditos
- **Desarrollador:** Carlos Quezada
- Inspirado en el trabajo de Alvaro Valenzuela
- Utiliza la funcionalidad de DAOPHOT desarrollada por Peter Stetson

## Uso
El programa se ejecuta mediante el script principal `daofun.py`. Proporciona una interfaz gráfica intuitiva para realizar diversas tareas de procesamiento de imágenes astronómicas.

### Dependencias
El proyecto depende de las siguientes bibliotecas y herramientas:
- `pandas`
- `matplotlib`
- `astropy`
- `aplpy`
- `IPython`
- `PySimpleGUI`

Las siguientes librerias fueron desarrolladas/acondicionadas durante el proyecto
- `daophot_wraps`
- `daofun_gui_selection`
- `daofun_backend`
- `fits_handler`

### Instalación
1. Clona este repositorio.
2. Asegúrate de tener todas las dependencias instaladas, puedes ejecutar `setup.py`.
4. Ejecuta `daofun.py` para iniciar la interfaz gráfica.

## Instrucciones de Uso
1. Ejecuta el programa con `python daofun.py`.
2. Selecciona un archivo `.fits` para comenzar el procesamiento de imágenes astronómicas.
3. Utiliza los botones proporcionados para realizar acciones como encontrar objetos, realizar fotometría, refinar PSFs, entre otros.

## Contribuciones
¡Las contribuciones son bienvenidas! Si deseas contribuir, sigue estos pasos:
1. Haz un `fork` del repositorio.
2. Crea una nueva rama (`branch`): `git checkout -b feature/nueva-caracteristica`.
3. Realiza tus cambios y haz `commit`: `git commit -am 'Agrega nueva característica'`.
4. Haz un `push` a la rama: `git push origin feature/nueva-caracteristica`.
5. Abre un `pull request` en GitHub.

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para obtener más detalles.

