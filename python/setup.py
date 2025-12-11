"""
TurboCat: Next-Generation Gradient Boosting Framework

Setup script for Python bindings.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build TurboCat. "
                "Install with: brew install cmake (macOS) or apt install cmake (Linux)"
            )
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        cfg = "Debug" if self.debug else "Release"
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DTURBOCAT_BUILD_PYTHON=ON",
            "-DTURBOCAT_BUILD_TESTS=OFF",
            "-DTURBOCAT_BUILD_BENCHMARKS=OFF",
        ]
        
        build_args = ["--config", cfg]
        
        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["--parallel"]
        
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp
        )


# Read README
this_dir = Path(__file__).parent
readme_path = this_dir.parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "TurboCat: Next-Generation Gradient Boosting Framework"


setup(
    name="turbocat",
    version="0.1.0",
    author="TurboCat Contributors",
    author_email="turbocat@example.com",
    description="Next-Generation Gradient Boosting Framework - Outperforms CatBoost on model quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ispromadhka/Turbo-Cat",
    
    packages=find_packages(),
    ext_modules=[CMakeExtension("turbocat._turbocat", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
    
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "scikit-learn>=1.0",
            "pandas>=1.3",
            "catboost>=1.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning gradient-boosting xgboost catboost lightgbm",
    
    project_urls={
        "Bug Reports": "https://github.com/ispromadhka/Turbo-Cat/issues",
        "Source": "https://github.com/ispromadhka/Turbo-Cat",
    },
)
