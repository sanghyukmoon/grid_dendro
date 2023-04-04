# GRID-dendro - Gravitational Identification with Dendrogram

### WARNING: This package is under construction 🏗

### References and Historical Remarks
- [Gong & Ostriker (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...729..120G/abstract): GRID
- [Mao, Ostriker, & Kim (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...52M/abstract): FISO

Gong & Ostriker (2011) developed and introduced a core finding method called GRID (GRavitatianal IDentification), which utilize isocontours of gravitational potential to provide physically motivated definition of bound cores. Mao, Ostriker, & Kim (2020) extended this algorithm to identify hierarchical structures of the interstellar medium using dendrogram, as well as significantly improve the efficiency of the algorithm. The first author of Mao, Ostriker, & Kim (2020) wrote a python pacakge [FISO](https://github.com/alwinm/fiso) (fast isocontours) to implement this algorithm.

`grid_dendro` is an implementation of an algorithm presented in Mao, Ostriker, & Kim (2020), written by refactoring the original [FISO](https://github.com/alwinm/fiso) code base to enhance readability and maintainability.

### Usage
