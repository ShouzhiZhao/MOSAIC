"""Keep explicitly private local modules out of wheels built from a working checkout."""

from setuptools import setup
from setuptools.command.build_py import build_py

LOCAL_MODULES = {
    "mosaic.caso.adaptation",
    "mosaic.caso.training",
    "mosaic.caso.two_stage_training",
    "mosaic.promosim.collection",
    "mosaic.promosim.collection_worker",
}


class ReleaseBuild(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            entry
            for entry in super().find_package_modules(package, package_dir)
            if f"{entry[0]}.{entry[1]}" not in LOCAL_MODULES
            and not (entry[0] == "baselines" and entry[1].startswith("plot_"))
        ]


setup(cmdclass={"build_py": ReleaseBuild})
