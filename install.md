# build package and install

- Build

```sh
python3 setup.py sdist bdist_wheel
```

- install locally

```sh
pip install . -v
```

- all together

```sh
python3 setup.py sdist bdist_wheel & pip install . -v
```

# Show pip version

```sh
pip show SpectrogramUtils
```

# Uninstall

```sh
pip uninstall SpectrogramUtils -y
```

# Clean previous builds

```sh
rm -rf build & rm -rf dist & rm -rf src/*.egg-info
```

# Uninstall and clean

```sh
pip uninstall SpectrogramUtils -y & rm -rf build & rm -rf dist & rm -rf src/*.egg-info
```
