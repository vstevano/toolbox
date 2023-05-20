"""
This module implements an XRD pattern calculator.
"""

from six.moves import filter
from six.moves import map
from six.moves import zip

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "5/22/14"


from math import sin, cos, asin, pi, degrees, radians, ceil
import os

import numpy as np
import numpy.linalg as npl
import json

from pylada.periodic_table import find as find_specie

#from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

#XRD wavelengths in angstroms
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}

#with open(os.path.join(os.path.dirname(__file__),
#                       "atomic_scattering_params.json")) as f:
#    ATOMIC_SCATTERING_PARAMS = json.load(f)


ATOMIC_SCATTERING_PARAMS = {"Ru": [[4.358, 27.881], [3.298, 5.179], [1.323, 0.594], [0, 0]], "Re": [[5.695, 28.968], [4.74, 5.156], [2.064, 0.575], [0, 0]], "Ra": [[6.215, 28.382], [5.17, 5.002], [2.316, 0.562], [0, 0]], "Rb": [[4.776, 140.782], [3.859, 18.991], [2.234, 3.701], [0.868, 0.419]], "Rn": [[4.078, 38.406], [4.978, 11.02], [3.096, 2.355], [1.326, 0.299]], "Rh": [[4.431, 27.911], [3.343, 5.153], [1.345, 0.592], [0, 0]], "Be": [[1.25, 60.804], [1.334, 18.591], [0.36, 3.653], [0.106, 0.416]], "Ba": [[7.821, 117.657], [6.004, 18.778], [3.28, 3.263], [1.103, 0.376]], "Bi": [[3.841, 50.261], [4.679, 11.999], [3.192, 2.56], [1.363, 0.318]], "Bk": [[6.502, 28.375], [5.478, 4.975], [2.51, 0.561], [0, 0]], "Br": [[2.166, 33.899], [2.904, 10.497], [1.395, 2.041], [0.589, 0.307]], "H": [[0.202, 30.868], [0.244, 8.544], [0.082, 1.273], [0, 0]], "P": [[1.888, 44.876], [2.469, 13.538], [0.805, 2.642], [0.32, 0.361]], "Os": [[5.75, 28.933], [4.773, 5.139], [2.079, 0.573], [0, 0]], "Ge": [[2.447, 55.893], [2.702, 14.393], [1.616, 2.446], [0.601, 0.342]], "Gd": [[5.225, 29.158], [4.314, 5.259], [1.827, 0.586], [0, 0]], "Ga": [[2.321, 65.602], [2.486, 15.458], [1.688, 2.581], [0.599, 0.351]], "Pr": [[5.085, 28.588], [4.043, 5.143], [1.684, 0.581], [0, 0]], "Pt": [[5.803, 29.016], [4.87, 5.15], [2.127, 0.572], [0, 0]], "Pu": [[6.415, 28.836], [5.419, 5.022], [2.449, 0.561], [0, 0]], "C": [[0.731, 36.995], [1.195, 11.297], [0.456, 2.814], [0.125, 0.346]], "Pb": [[3.51, 52.914], [4.552, 11.884], [3.154, 2.571], [1.359, 0.321]], "Pa": [[6.306, 28.688], [5.303, 5.026], [2.386, 0.561], [0, 0]], "Pd": [[4.436, 28.67], [3.454, 5.269], [1.383, 0.595], [0, 0]], "Cd": [[2.574, 55.675], [3.259, 11.838], [2.547, 2.784], [0.838, 0.322]], "Po": [[6.07, 28.075], [4.997, 4.999], [2.232, 0.563], [0, 0]], "Pm": [[5.201, 28.079], [4.094, 5.081], [1.719, 0.576], [0, 0]], "Ho": [[5.376, 28.773], [4.403, 5.174], [1.884, 0.582], [0, 0]], "Hf": [[5.588, 29.001], [4.619, 5.164], [1.997, 0.579], [0, 0]], "Hg": [[2.682, 42.822], [4.241, 9.856], [2.755, 2.295], [1.27, 0.307]], "He": [[0.091, 18.183], [0.181, 6.212], [0.11, 1.803], [0.036, 0.284]], "Mg": [[2.268, 73.67], [1.803, 20.175], [0.839, 3.013], [0.289, 0.405]], "K": [[3.951, 137.075], [2.545, 22.402], [1.98, 4.532], [0.482, 0.434]], "Mn": [[2.747, 67.786], [2.456, 15.674], [1.792, 3.0], [0.498, 0.357]], "O": [[0.455, 23.78], [0.917, 7.622], [0.472, 2.144], [0.138, 0.296]], "S": [[1.659, 36.65], [2.386, 11.488], [0.79, 2.469], [0.321, 0.34]], "W": [[5.709, 28.782], [4.677, 5.084], [2.019, 0.572], [0, 0]], "Zn": [[1.942, 54.162], [1.95, 12.518], [1.619, 2.416], [0.543, 0.33]], "Eu": [[6.267, 100.298], [4.844, 16.066], [3.202, 2.98], [1.2, 0.367]], "Zr": [[4.105, 28.492], [3.144, 5.277], [1.229, 0.601], [0, 0]], "Er": [[5.436, 28.655], [4.437, 5.117], [1.891, 0.577], [0, 0]], "Ni": [[2.21, 58.727], [2.134, 13.553], [1.689, 2.609], [0.524, 0.339]], "Na": [[2.241, 108.004], [1.333, 24.505], [0.907, 3.391], [0.286, 0.435]], "Nb": [[4.237, 27.415], [3.105, 5.074], [1.234, 0.593], [0, 0]], "Nd": [[5.151, 28.304], [4.075, 5.073], [1.683, 0.571], [0, 0]], "Ne": [[0.303, 17.64], [0.72, 5.86], [0.475, 1.762], [0.153, 0.266]], "Np": [[6.323, 29.142], [5.414, 5.096], [2.453, 0.568], [0, 0]], "Fr": [[6.201, 28.2], [5.121, 4.954], [2.275, 0.556], [0, 0]], "Fe": [[2.544, 64.424], [2.343, 14.88], [1.759, 2.854], [0.506, 0.35]], "B": [[0.945, 46.444], [1.312, 14.178], [0.419, 3.223], [0.116, 0.377]], "F": [[0.387, 20.239], [0.811, 6.609], [0.475, 1.931], [0.146, 0.279]], "Sr": [[5.848, 104.972], [4.003, 19.367], [2.342, 3.737], [0.88, 0.414]], "N": [[0.572, 28.847], [1.043, 9.054], [0.465, 2.421], [0.131, 0.317]], "Kr": [[2.034, 29.999], [2.927, 9.598], [1.342, 1.952], [0.589, 0.299]], "Si": [[2.129, 57.775], [2.533, 16.476], [0.835, 2.88], [0.322, 0.386]], "Sn": [[3.45, 59.104], [3.735, 14.179], [2.118, 2.855], [0.877, 0.327]], "Sm": [[5.255, 28.016], [4.113, 5.037], [1.743, 0.577], [0, 0]], "V": [[3.245, 76.379], [2.698, 17.726], [1.86, 3.363], [0.486, 0.374]], "Sc": [[3.966, 88.96], [2.917, 20.606], [1.925, 3.856], [0.48, 0.399]], "Sb": [[3.564, 50.487], [3.844, 13.316], [2.687, 2.691], [0.864, 0.316]], "Se": [[2.298, 38.83], [2.854, 11.536], [1.456, 2.146], [0.59, 0.316]], "Co": [[2.367, 61.431], [2.236, 14.18], [1.724, 2.725], [0.515, 0.344]], "Cm": [[6.46, 28.396], [5.469, 4.97], [2.471, 0.554], [0, 0]], "Cl": [[1.452, 30.935], [2.292, 9.98], [0.787, 2.234], [0.322, 0.323]], "Ca": [[4.47, 99.523], [2.971, 22.696], [1.97, 4.195], [0.482, 0.417]], "Cf": [[6.548, 28.461], [5.526, 4.965], [2.52, 0.557], [0, 0]], "Ce": [[5.007, 28.283], [3.98, 5.183], [1.678, 0.589], [0, 0]], "Xe": [[3.366, 35.509], [4.147, 11.117], [2.443, 2.294], [0.829, 0.289]], "Tm": [[5.441, 29.149], [4.51, 5.264], [1.956, 0.59], [0, 0]], "Cs": [[6.062, 155.837], [5.986, 19.695], [3.303, 3.335], [1.096, 0.379]], "Cr": [[2.307, 78.405], [2.334, 15.785], [1.823, 3.157], [0.49, 0.364]], "Cu": [[1.579, 62.94], [1.82, 12.453], [1.658, 2.504], [0.532, 0.333]], "La": [[4.94, 28.716], [3.968, 5.245], [1.663, 0.594], [0, 0]], "Li": [[1.611, 107.638], [1.246, 30.48], [0.326, 4.533], [0.099, 0.495]], "Tl": [[5.932, 29.086], [4.972, 5.126], [2.195, 0.572], [0, 0]], "Lu": [[5.553, 28.907], [4.58, 5.16], [1.969, 0.577], [0, 0]], "Th": [[6.264, 28.651], [5.263, 5.03], [2.367, 0.563], [0, 0]], "Ti": [[3.565, 81.982], [2.818, 19.049], [1.893, 3.59], [0.483, 0.386]], "Te": [[4.785, 27.999], [3.688, 5.083], [1.5, 0.581], [0, 0]], "Tb": [[5.272, 29.046], [4.347, 5.226], [1.844, 0.585], [0, 0]], "Tc": [[4.318, 28.246], [3.27, 5.148], [1.287, 0.59], [0, 0]], "Ta": [[5.659, 28.807], [4.63, 5.114], [2.014, 0.578], [0, 0]], "Yb": [[5.529, 28.927], [4.533, 5.144], [1.945, 0.578], [0, 0]], "Dy": [[5.332, 28.888], [4.37, 5.198], [1.863, 0.581], [0, 0]], "I": [[3.473, 39.441], [4.06, 11.816], [2.522, 2.415], [0.84, 0.298]], "U": [[6.767, 85.951], [6.729, 15.642], [4.014, 2.936], [1.561, 0.335]], "Y": [[4.129, 27.548], [3.012, 5.088], [1.179, 0.591], [0, 0]], "Ac": [[6.278, 28.323], [5.195, 4.949], [2.321, 0.557], [0, 0]], "Ag": [[2.036, 61.497], [3.272, 11.824], [2.511, 2.846], [0.837, 0.327]], "Ir": [[5.754, 29.159], [4.851, 5.152], [2.096, 0.57], [0, 0]], "Am": [[6.378, 29.156], [5.495, 5.102], [2.495, 0.565], [0, 0]], "Al": [[2.276, 72.322], [2.428, 19.773], [0.858, 3.08], [0.317, 0.408]], "As": [[2.399, 45.718], [2.79, 12.817], [1.529, 2.28], [0.594, 0.328]], "Ar": [[1.274, 26.682], [2.19, 8.813], [0.793, 2.219], [0.326, 0.307]], "Au": [[2.388, 42.866], [4.226, 9.743], [2.689, 2.264], [1.255, 0.307]], "At": [[6.133, 28.047], [5.031, 4.957], [2.239, 0.558], [0, 0]], "In": [[3.153, 66.649], [3.557, 14.449], [2.818, 2.976], [0.884, 0.335]], "Mo": [[3.12, 72.464], [3.906, 14.642], [2.361, 3.237], [0.85, 0.366]]}
####

class XRDCalculator(object):
    """
    Computes the XRD pattern of a crystal structure.
    This code is implemented by Shyue Ping Ong as part of UCSD's NANO106 -
    Crystallography of Materials. The formalism for this code is based on
    that given in Chapters 11 and 12 of Structure of Materials by Marc De
    Graef and Michael E. McHenry. This takes into account the atomic
    scattering factors and the Lorentz polarization factor, but not
    the Debye-Waller (temperature) factor (for which data is typically not
    available). Note that the multiplicity correction is not needed since
    this code simply goes through all reciprocal points within the limiting
    sphere, which includes all symmetrically equivalent planes. The algorithm
    is as follows
    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.
    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`
    3. Compute the structure factor as the sum of the atomic scattering
       factors. The atomic scattering factors are given by
       .. math::
           f(s) = Z - 41.78214 \\times s^2 \\times \\sum\\limits_{i=1}^n a_i \
           \exp(-b_is^2)
       where :math:`s = \\frac{\\sin(\\theta)}{\\lambda}` and :math:`a_i`
       and :math:`b_i` are the fitted parameters for each element. The
       structure factor is then given by
       .. math::
           F_{hkl} = \\sum\\limits_{j=1}^N f_j \\exp(2\\pi i \\mathbf{g_{hkl}}
           \cdot \\mathbf{r})
    4. The intensity is then given by the modulus square of the structure
       factor.
       .. math::
           I_{hkl} = F_{hkl}F_{hkl}^*
    5. Finally, the Lorentz polarization correction factor is applied. This
       factor is given by:
       .. math::
           P(\\theta) = \\frac{1 + \\cos^2(2\\theta)}
           {\\sin^2(\\theta)\\cos(\\theta)}
    """

    #Tuple of available radiation keywords.
    AVAILABLE_RADIATION = tuple(WAVELENGTHS.keys())

    #Tolerance in which to treat two peaks as having the same two theta.
    TWO_THETA_TOL = 1e-5

    # Tolerance in which to treat a peak as effectively 0 if the scaled
    # intensity is less than this number. Since the max intensity is 100,
    # this means the peak must be less than 1e-5 of the peak intensity to be
    # considered as zero. This deals with numerical issues where systematic
    # absences do not cancel exactly to zero.
    SCALED_INTENSITY_TOL = 1e-3

    def __init__(self, wavelength="CuKa", symprec=0, debye_waller_factors=None):
        """
        Initializes the XRD calculator with a given radiation.
        Args:
            wavelength (str/float): The wavelength can be specified as either a
                float or a string. If it is a string, it must be one of the
                supported definitions in the AVAILABLE_RADIATION class
                variable, which provides useful commonly used wavelengths.
                If it is a float, it is interpreted as a wavelength in
                angstroms. Defaults to "CuKa", i.e, Cu K_alpha radiation.
            symprec (float): Symmetry precision for structure refinement. If
                set to 0, no refinement is done. Otherwise, refinement is
                performed using spglib with provided precision.
                NOTE: NREL version ignores this param!
            debye_waller_factors ({element symbol: float}): Allows the
                specification of Debye-Waller factors. Note that these
                factors are temperature dependent.
        """
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        self.symprec = symprec
        self.debye_waller_factors = debye_waller_factors or {}

    def get_xrd_data(self, structure, scaled=True, two_theta_range=(0, 90)):
        """
        Calculates the XRD data for a structure.
        Args:
            structure (Structure): Input structure
            scaled (bool): Whether to return scaled intensities. The maximum
                peak is set to a value of 1. Defaults to True. Use False if
                you need the absolute values to combine XRD plots.
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
        Returns:
            (XRD pattern) in the form of
            [[two_theta, intensity, {(h, k, l): mult}, d_hkl], ...]
            Two_theta is in degrees. Intensity is in arbitrary units and if
            scaled (the default), has a maximum value of 100 for the highest
            peak. {(h, k, l): mult} is a dict of Miller indices for all
            diffracted lattice planes contributing to that intensity and
            their multiplicities. d_hkl is the interplanar spacing.
        """
##        if self.symprec:
##            finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
##            structure = finder.get_refined_structure()

        structure.cell = structure.cell * structure.scale
        for ii in range(len(structure)):
            structure[ii].pos *= structure.scale
        structure.scale = 1.

        wavelength = self.wavelength
##        latt = structure.lattice
##        is_hex = latt.is_hexagonal()
        latt = structure.cell
        invlatt = npl.inv(latt)

        # Obtained from Bragg condition. Note that reciprocal lattice
        # vector length is 1 / d_hkl.
        min_r, max_r = (0, 2 / wavelength) if two_theta_range is None else \
            [2 * sin(radians(t / 2)) / wavelength for t in two_theta_range]

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt=invlatt
##        recip_latt = latt.reciprocal_lattice_crystallographic
##        recip_pts = recip_latt.get_points_in_sphere(
##            [[0, 0, 0]], [0, 0, 0], max_r)
        recip_pts = get_points_in_sphere(recip_latt, max_r)

#        print "#######################"
#        print recip_latt
#        print min_r, max_r
#        for rps in recip_pts:
#            print rps
#        print "#######################"
        
        if min_r:
            recip_pts = filter(lambda d: d[1] >= min_r, recip_pts)

        # Create a flattened array of zs, coeffs, fcoords and occus. This is
        # used to perform vectorized computation of atomic scattering factors
        # later. Note that these are not necessarily the same size as the
        # structure as each partially occupied specie occupies its own
        # position in the flattened array.
        zs = []
        coeffs = []
        fcoords = []
        occus = []
        dwfactors = []

        for site in structure:
            anum = find_specie(name=site.type).atomic_number
#            print "site ", site, site.type, anum
#            for sp, occu in site.species_and_occu.items():
#                print "sp, occu", sp, occu
            zs.append(anum)
            try:
                c = ATOMIC_SCATTERING_PARAMS[site.type]
            except KeyError:
                raise ValueError("Unable to calculate XRD pattern as "
                                 "there is no scattering coefficients for"
                                 " %s." % sp.symbol)
            coeffs.append(c)
            dwfactors.append(self.debye_waller_factors.get(site.type, 0))
            fc = np.dot(invlatt,site.pos)
#            print "frac coords", fc
            fcoords.append(fc)
#            occus.append(occu)
            occus.append(1)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)
        dwfactors = np.array(dwfactors)
        peaks = {}
        two_thetas = []

        for hkl, g_hkl, ind in sorted(
                recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])):
            
            if g_hkl != 0:

                d_hkl = 1 / g_hkl

                # Bragg condition
                theta = asin(wavelength * g_hkl / 2)

                # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
                # 1/|ghkl|)
                s = g_hkl / 2

                #Store s^2 since we are using it a few times.
                s2 = s ** 2

                # Vectorized computation of g.r for all fractional coords and
                # hkl.
                g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

                # Highly vectorized computation of atomic scattering factors.
                # Equivalent non-vectorized code is::
                #
                #   for site in structure:
                #      el = site.specie
                #      coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
                #      fs = el.Z - 41.78214 * s2 * sum(
                #          [d[0] * exp(-d[1] * s2) for d in coeff])
                fs = zs - 41.78214 * s2 * np.sum(
                    coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)

                dw_correction = np.exp(-dwfactors * s2)

                # Structure factor = sum of atomic scattering factors (with
                # position factor exp(2j * pi * g.r and occupancies).
                # Vectorized computation.
                f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r)
                               * dw_correction)

                #Lorentz polarization correction for hkl
                lorentz_factor = (1 + cos(2 * theta) ** 2) / \
                    (sin(theta) ** 2 * cos(theta))

                # Intensity for hkl is modulus square of structure factor.
                i_hkl = (f_hkl * f_hkl.conjugate()).real

                two_theta = degrees(2 * theta)

##                if is_hex:
                    #Use Miller-Bravais indices for hexagonal lattices.
##                    hkl = (hkl[0], hkl[1], - hkl[0] - hkl[1], hkl[2])
                #Deal with floating point precision issues.
                ind = np.where(np.abs(np.subtract(two_thetas, two_theta)) <
                               XRDCalculator.TWO_THETA_TOL)
                if len(ind[0]) > 0:
                    peaks[two_thetas[int(ind[0])]][0] += i_hkl * lorentz_factor
                    peaks[two_thetas[int(ind[0])]][1].append(tuple(hkl))
                else:
                    peaks[two_theta] = [i_hkl * lorentz_factor, [tuple(hkl)],
                                        d_hkl]
                    two_thetas.append(two_theta)

        # Scale intensities so that the max intensity is 1.
        max_intensity = max([v[0] for v in peaks.values()])
        data = []
        for k in sorted(peaks.keys()):
            v = peaks[k]
            scaled_intensity = v[0] / max_intensity * 1.0 if scaled else v[0]
            fam = get_unique_families(v[1])
            if scaled_intensity > XRDCalculator.SCALED_INTENSITY_TOL:
                data.append([k, scaled_intensity, fam, v[2]])
        return data


    # Vladan
    def broadened(self,strc,sigma=2,two_theta_range=(0, 180)):
        """
        function that returns gaussian broadened 
        and normalized (integral=1) XRD pattern
        """

        # just a gaussian
        def gauss(x,x0,sigma):
            return 1./np.sqrt(2*np.pi*sigma**2) * np.e**( - (x-x0)**2/(2*sigma**2)   )
    
        # getting the xrd
        xrd_data=self.get_xrd_data(strc, scaled=True, two_theta_range=two_theta_range)

        # grid on the 2theta axis
        delta = sigma/float(10)
        angles = np.arange(two_theta_range[0], two_theta_range[1]+delta, delta)
        
        # placing a gaussian on every XRD peak
        data = np.zeros(len(angles))
        
        for xd in xrd_data:
            data = data + float(xd[1])*gauss(angles,xd[0],sigma)

        data = data/np.sum(data)/delta

#        return zip(angles,data)
        return np.array([[angles[ii],data[ii]] for ii in range(len(angles))])


def get_points_in_sphere(latt, r):
    """
    Find all points within a sphere from the point taking into account
    periodic boundary conditions. This includes sites in other periodic
    images.
    Algorithm:
    1. place sphere of radius r in crystal and determine minimum supercell
       (parallelpiped) which would contain a sphere of radius r. for this
       we need the projection of a_1 on a unit vector perpendicular
       to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
       determine how many a_1"s it will take to contain the sphere.
       Nxmax = r * length_of_b_1 / (2 Pi)
    2. keep points falling within r.
    Args:
        frac_points: All points in the lattice in fractional coordinates.
        center: Cartesian coordinates of center of sphere.
        r: radius of sphere.
    Returns:
        [(fcoord, dist) ...] since most of the time, subsequent processing
        requires the distance.
    """
    import math

    frac_points = [[0,0,0]]
    center = [0,0,0]

    rl = 2*pi*npl.inv(latt)
    recp_len = np.array([npl.norm(rl[:,i]) for i in range(3)])
#    print "recp_len", recp_len
#    recp_len = np.array(self.Reciprocal_lattice.abc)
    sr = r + 0.15
    nmax = sr * recp_len / (2 * math.pi)
#    pcoords = self.get_fractional_coords(center)
    pcoords = np.array([0,0,0])
    floor = math.floor

    n = len(frac_points)
    fcoords = np.array(frac_points)
    pts = np.tile(center, (n, 1))
    indices = np.array(list(range(n)))

    arange = np.arange(start=int(floor(pcoords[0] - nmax[0])),
                       stop=int(floor(pcoords[0] + nmax[0])) + 1)
    brange = np.arange(start=int(floor(pcoords[1] - nmax[1])),
                       stop=int(floor(pcoords[1] + nmax[1])) + 1)
    crange = np.arange(start=int(floor(pcoords[2] - nmax[2])),
                       stop=int(floor(pcoords[2] + nmax[2])) + 1)

    arange = arange[:, None] * np.array([1, 0, 0])[None, :]
    brange = brange[:, None] * np.array([0, 1, 0])[None, :]
    crange = crange[:, None] * np.array([0, 0, 1])[None, :]

    images = arange[:, None, None] + brange[None, :, None] +\
        crange[None, None, :]

    shifted_coords = fcoords[:, None, None, None, :] + images[None, :, :, :, :]
 #   coords = self.get_cartesian_coords(shifted_coords)
    coords = np.dot(shifted_coords, latt)
    dists = np.sqrt(np.sum((coords - pts[:, None, None, None, :]) ** 2,
                           axis=4))
    within_r = np.where(dists <= r)

    return list(zip(shifted_coords[within_r], dists[within_r],
                    indices[within_r[0]]))


def get_unique_families(hkls):
    """
    Returns unique families of Miller indices. Families must be permutations
    of each other.
    Args:
        hkls ([h, k, l]): List of Miller indices.
    Returns:
        {hkl: multiplicity}: A dict with unique hkl and multiplicity.
    """
    #TODO: Definitely can be sped up.
    def is_perm(hkl1, hkl2):
        h1 = map(abs, hkl1)
        h2 = map(abs, hkl2)
        return all([i == j for i, j in zip(sorted(h1), sorted(h2))])

    unique = {}
    for hkl1 in hkls:
        found = False
        for hkl2 in unique.keys():
            if is_perm(hkl1, hkl2):
                found = True
                unique[hkl2] += 1
                break
        if not found:
            unique[hkl1] = 1

    return unique
########################

## Structure factor 
## Does not distinguish atomic species (this version)

def strc_factor(strc=None,q=None):
    '''
    strc: pylada Structure
    q: q-vestor
    '''
    return np.sum([np.exp(-1j*np.dot(q,atom.pos*strc.scale)) for atom in strc])


