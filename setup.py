from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autogen-voting-extension",
    version="0.1.0",
    author="Tejas Dharani",
    author_email="tejas@example.com",  # Update with your email
    description="Enhanced voting-based group chat for AutoGen with multiple voting strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tejas-dharani/autogen-voting-extension",  # Update with your repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "autogen-agentchat>=0.4.0",
        "autogen-core>=0.4.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
)
