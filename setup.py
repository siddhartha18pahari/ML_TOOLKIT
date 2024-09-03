import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "MLV_toolbox_python",
    version = "0.0.4",
    author = "Dirk Bernhardt-Walther, Sadman Hossain",
    author_email = "m.sadman.h@gmail.com",
    description = "A tool for researchers to extract structural properties of contours such as orientation, length, curvature and junctions present in images",
    url = "https://github.com/sadmanca/MLV_toolbox_python",
    project_urls = {
        "Roadmap": "https://github.com/users/sadmanca/projects/4",
        "Issues": "https://github.com/sadmanca/MLV_toolbox_python/issues",
    }, 
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "MLV_toolbox_python"},
    packages = setuptools.find_packages(where="MLV_toolbox_python"),
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python",
        "tabulate",
    ]
)
