from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='SpectrogramUtils',
        version='0.1',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        install_requires=[
            # List your project's dependencies here
            # e.g., 'requests', 'numpy',
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