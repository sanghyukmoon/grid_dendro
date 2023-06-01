# GRID-Dendro -- Gravitational Identification with Dendrogram

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://sanghyukmoon.github.io/grid_dendro)


## Acknowledgments and Historical Remarks
{cite:t}`GongOstriker2011` developed and introduced a core finding method called GRID (GRavitatianal IDentification), which utilize isocontours of gravitational potential to provide physically motivated definition of bound cores. {cite:t}`Mao2020` extended this algorithm to identify hierarchical structures of the interstellar medium using dendrogram, as well as significantly improve the efficiency of the algorithm. Alwin Mao, who is the first author of {cite:t}`Mao2020`, wrote a python pacakge [FISO](https://github.com/alwinm/fiso) (fast isocontours) to implement this algorithm in the course of his PhD program. The code has an optimal performance thanks to Alwin's huge effort to make it fast. Because FISO is so good, I wanted to make it more user-friendly in terms of code design and documentation. Alwin kindly agreed that I can refactor the code and maintain it in a [separate repository](https://github.com/sanghyukmoon/grid_dendro).


## Example Usage
```
>>> import pyathena as pa
>>> s = pa.LoadSim("R8_2pc")
>>> ds = s.load_vtk(200)
>>> dat = ds.get_field(['density','velocity','pressure'])
>>> ds_grav_pp = pa.read_vtk("R8_2pc/phi_gas_only/R8_2pc.0200.Phi.vtk")
>>> dat['gravitational_potential'] = ds_grav_pp.get_field('Phi').Phi
>>> gd = dendrogram.Dendrogram(ds.gravitational_potential.data)
>>> gd.construct()
>>> gd.prune()  # Remove buds
>>> data = dict(rho=dat.density.data,
                vel1=dat.velocity1.data,
                vel2=dat.velocity2.data,
                vel3=dat.velocity3.data,
                prs=dat.pressure.data,
                phi=dat.gravitational_potential.data,
                dvol=s.domain['dx'].prod())
>>> hbp, hbr = find_bound_object(gd, data)
```


## References
```{bibliography}
```
