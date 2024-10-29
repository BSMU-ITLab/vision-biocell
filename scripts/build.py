"""
See build instructions:
https://github.com/BSMU-ITLab/vision/blob/main/scripts/build.py
"""

from pathlib import Path

import bsmu.biocell
from bsmu.biocell.app import BiocellApp
from bsmu.vision.app.builder import AppBuilder


if __name__ == '__main__':
    app_builder = AppBuilder(
        project_dir=Path(__file__).resolve().parents[1],
        app_class=BiocellApp,

        script_path_relative_to_project_dir=Path('src/bsmu/biocell/app/__main__.py'),
        icon_path_relative_to_project_dir=Path('src/bsmu/biocell/app/images/icons/biocell.ico'),

        add_packages=['bsmu.biocell', 'scipy.optimize', 'scipy.integrate'],
        add_packages_with_data=[bsmu.biocell],
    )
    app_builder.build()
