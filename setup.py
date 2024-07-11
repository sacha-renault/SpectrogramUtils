from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='SpectrogramUtils',
        use_scm_version=True,
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        install_requires=[
            "numpy",
            "librosa",
            "soundfile",
            "matplotlib",
            "scikit-learn"
        ],
        author='Sacha Renault',
        author_email='',
        description='A brief description of your module',
        long_description=open('readme.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/sacha-renault/SpectrogramUtils',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.6',
    )