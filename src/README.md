# BoreFlow

The Python package BoreFlow provides a simple numerical model using Steep-Slope Shallow Water Equations to describe the flow of overtopping waves and bores.

This Python package is developed as part of the Ph.D. research of Niels van der Vegt and is published under the GNU GPL-3 license.

## Getting started

To download the package run `pip install boreflow`

```py
import numpy as np

from boreflow import BCArray, Geometry, Simulation

# 1) Create geometry
geometry = Geometry([0, 2, 11], [3, 3, 0], [0.0175, 0.0175])

# 2) Create boundary conditions
t = np.array([0, 1, 5])
h = np.array([0.5, 0.8, 0])
u = np.array([1.0, 2.0, 0])
bc = BCArray(t, h, u)

# 3) Initialize simulation settings
sim = Simulation(t_end=10.0, cfl=0.2, max_dt=0.01, nx=110)

# 4) Run the simulation
results = sim.run(geometry, bc)

# 5) Analyse the flow, e.g. at s=10m
res_t, res_h, res_u = results.get_st(s=10.0)
```

## Acknowledgements

The authors would like to thank the researchers who have conducted studies on overtopping flow. All studies are referenced in the code of the respective boundary condition implementations. Furthermore, we acknowledge the publication by [Maranzoni and Tomirotti (2022)](https://doi.org/10.1016/j.advwatres.2022.104255) for their publication regarding the Steep-Slope Shallow Water Equations.

This work is part of the Perspectief research programme Future Flood Risk Management Technologies for Rivers and Coasts with project number P21-23. This programme is financed by Domain Applied and Engineering Sciences of the Dutch Research Council (NWO).

HKV Lijn in Water is acknowledged for their [Python package template](https://github.com/HKV-products-services/python_package_template), which is used to publish this package.