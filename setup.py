from setuptools import setup, find_packages

setup(
    name="qclaw",
    version="0.1.0",
    description="Quantum-Classical Logic & Action Wrapper — Route optimization to QPUs",
    author="GPUPulse",
    author_email="adjusternwachukwu@gmail.com",
    url="https://github.com/Quantum-bridge-labs/qclaw",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "originq": ["pyqpanda>=3.8"],
        "server": ["fastapi", "uvicorn"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
