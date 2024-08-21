"""
See build instructions:
https://github.com/BSMU-ITLab/vision/blob/main/scripts/build.py
"""


from pathlib import Path

import bsmu.biocell.app
import bsmu.biocell.plugins
from bsmu.vision.app.builder import AppBuilder

if __name__ == '__main__':
    app_builder = AppBuilder(
        project_dir=Path(__file__).resolve().parents[1],
        script_path_relative_to_project_dir=Path('src/bsmu/biocell/app/__main__.py'),

        app_name=bsmu.biocell.app.__title__,
        app_version=bsmu.biocell.app.__version__,
        app_description=bsmu.biocell.app.__description__,
        icon_path_relative_to_project_dir=Path('src/bsmu/biocell/app/images/icons/biocell.ico'),

        add_packages=['bsmu.biocell.app', 'bsmu.biocell.plugins', 'scipy.optimize', 'scipy.integrate'],
        add_packages_with_data=[bsmu.biocell.app, bsmu.biocell.plugins],
    )
    app_builder.build()
