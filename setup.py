from setuptools import setup, find_packages

# Read requirements (skip blank lines, comments and pip options such as --index-url)
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    ]

# Read description (for PyPI/GitHub)
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="efficient-diffusion-loader",
    version="0.1.0",
    author="Yash Sinha",
    description="Optimize Stable Diffusion inference on consumer hardware with Fractional Batching.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yashsinha1702/edgeforge-ai",
    
    # This automatically finds your code inside 'src'
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    install_requires=requirements,
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)