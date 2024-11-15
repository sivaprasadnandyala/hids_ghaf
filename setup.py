from setuptools import setup, find_packages

setup(
    name='hids',       
    version='0.0.1',
    packages=find_packages(where='./'),  # Automatically discover packages under src
    package_dir={'': './'},  # Tell setuptools where the packages are located
    package_data={
        'src': [
            'config.yaml',
            'syscallpolicy.yaml',
            'models/checkpoints/*',  # All files in models/checkpoints
            'models/final/*',        # All files in models/final
            'models/scaler.pkl',     # File in models/
        ],
    },
    include_package_data=True,
    install_requires=[
        #'pytorch',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'seaborn',
        # Add other dependencies if needed
    ],
    entry_points={
        'console_scripts': [
            'rtinference=src.realtime_inference:main',
            'datagather=src.data_gathering:main',
        ],
    },
    author='XXX',
    author_email='xxx.yy@tii.ae',
    description='A Python package to generate secure configuration for systemd service.',
    long_description=open('README.md').read(),
    url='https://github.com/xx/yyy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
