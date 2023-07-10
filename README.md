# toolbox
Collection of tools for pylada

Dependencies: pylada, spglib, scipy

The idea is to to put all the auxiliary codes we wrote over time in one place. The toolbox folder needs to be copied into the corresponding python site-packages or in some place visible via the PYTHONPATH variable.

The list of tools is:

1. cut_slab.py

A class constructed to cut slabs of a given thickness from a given bulk structure for an arbitrary set of Miller indices hkl. There are also functions to produce terminations that minimize: (a) the number of broken bonds of surface atoms, (b) surogate surface energy, and (c) Madelung energy, and to compute the dipole moment in the hkl direction from the atomic positions and the oxidation states as charges Returns a slab from a 3D pylada structure.

2. RSL_gen.py

Generator of random supperlattice (RSL) structures for ab-initio random structure sampling. For details please refer to: V. Stevanovic, Phys. Rev. Lett. 116, 075503 (2016) DOI:http://dx.doi.org/10.1103/PhysRevLett.116.075503. The goal is to generate random structures that favor cation-anion coordination. This is achieved by distributing different kinds of atoms randomly over two interpenetrating grids of points. The grids are constructed using planes of a supperlattice defined by a randomly chosen reciprocal lattice vector.

3. wulff.py

Class to do Wulff construction from surface energies and compute various other relevant quantities given the crystal structure and surface energies.

4. format_spglib.py

Functions to translate pylada structure object into the spglib one and back

5. xrdcalculator.py

A class to compute the powder diffraction pattern for a given structure. Originally written by Shuye Ping Ong for pymatgen and adapted for pylada by P. Graf and later V. Stevanovic (see https://materialsproject.org/about/terms)

6. pdf.py

Function that computes partial pair distribution functions (atom type resovled) between 0. and cutoff_r with the grid on the distance axis defined by dr. Returns a dictionary with keys labeling the types of atoms for which g(r) is calculated (e.g. 'Si-Si','Si-O', etc.) and the values are corresponding g(r).

7. fancy_firstshell.py

A function that uses the original pylada neighbors function to compute the first shell coordination and neighbors of atoms. It still uses the original neighbors function, just makes sure the output fullfils additional criteria.

8. solid_angle.py

Function to evaluate solid angle spanned by a polygon as viewed from a center point outside of it. This is done by splitting the pyramid into tetrahedra and then summing all solid angles of individual tetrahedra.

9. mean_value_point.py

A code to compute the Mean Value (Baldereschi's) Point of a given structure in the first Brillouin zone. For a detailed explanation of the fundamentals of the mean value point and how to compute it please check: A. Baldcreschi, 'Mean-Value Point in the Brillouin Zone', Phys. Rev. B 7, 5212 (1973) DOI: https://doi.org/10.1103/PhysRevB.7.5212 .
This code follows directly the reasonong from Baldereschi, with the caveat that the system of equations that needs to be solved to obtain the mean value point is solved numerically on the grid of k-points.

