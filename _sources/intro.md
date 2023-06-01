# GRID-Dendro -- Gravitational Identification with Dendrogram

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://sanghyukmoon.github.io/grid_dendro)


## Acknowledgments and Historical Remarks
{cite:t}`GongOstriker2011` developed and introduced a core finding method called GRID (GRavitatianal IDentification), which utilize isocontours of gravitational potential to provide physically motivated definition of bound cores. {cite:t}`Mao2020` extended this algorithm to identify hierarchical structures of the interstellar medium using dendrogram, as well as significantly improve the efficiency of the algorithm. Alwin Mao, who is the first author of {cite:t}`Mao2020`, wrote a python pacakge [FISO](https://github.com/alwinm/fiso) (fast isocontours) to implement this algorithm in the course of his PhD program. The code has an optimal performance thanks to Alwin's huge effort to make it fast. Because FISO is so good, I wanted to make it more user-friendly in terms of code design and documentation. Alwin kindly agreed that I can refactor the code and maintain it in a [separate repository](https://github.com/sanghyukmoon/grid_dendro).


## Example Usage
```
>>> import pyathena as pa
>>> s = pa.LoadSim("/scratch/smoon/M5J2P0N512")
>>> ds = s.load_hdf5(50, load_method='pyathena')
>>> cs = 1.0 # isothermal sound speed
>>> gd = dendrogram.Dendrogram(ds.phigas.data)
>>> gd.construct()
>>> gd.prune()  # Remove buds
>>> data = dict(rho=ds.dens.data,
                vel1=(ds.mom1/ds.dens).data,
                vel2=(ds.mom2/ds.dens).data,
                vel3=(ds.mom3/ds.dens).data,
                prs=(cs**2*ds.dens).data,
                phi=ds.phigas.data,
                dvol=s.domain['dx'].prod())
>>> hbp, hbr = find_bound_object(gd, data)
```


## References
```{bibliography}
```
