from setuptools import setup, find_packages
import os
from glob import glob

package_name = "vinpro_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VinPRO Team",
    maintainer_email="vinpro@polito.it",
    description="WP2 ViNet inference bridge node for VinPRO",
    license="MIT",
    entry_points={
        "console_scripts": [
            "inference_node = vinpro_perception.inference_node:main",
        ],
    },
)
