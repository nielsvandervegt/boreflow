# Python_package_template

This template package contains core functionality for a python package, developed to showcase and speed up the process of developing a python package. This Python package template is developed by HKV and is published under the GNU GPL-3 license.

## Getting started

### Using install (in future)

run `pip install Python_package_template`

### developing with pixi

To manage the environment we use Pixi.

#### windows

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### Linux/Mac

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

#### installing

With the `Pixi` command in powershell install the python environment:

```bash
 cd ../python_package_template
 pixi install
```

The `pixi.lock` file loads the correct packages and downloads to the `.pixi` file, you can use this environment in developing and resting.

For questions about how to use this package contact `dupuits@hkv.nl` or `haasnoot@hkv.nl`.
