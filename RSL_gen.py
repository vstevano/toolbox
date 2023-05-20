###############################
#
#  This file is written by Vladan Stevanovic
#
###############################

from pylada.crystal import Structure,neighbors
from random import random, randint,choice
from numpy import pi,exp,sqrt,cos,sin,dot,array,transpose,zeros,arange
from numpy.linalg import inv, det

########################################

def prob(x,width,center,structure):
    """
    Just an auxiliary function that helps with creating the 
    probability distribution that is utilized to ensure 
    homogeneous distribution of ions on the grid. Basically 
    returns a value of a Gaussian (not normalized, becuase 
    it does not matter) centered on a given point x inside 
    the unit cell (center) accounting also for periodic 
    images of the center

    :param x: 3x1 numpy array
         position vector of a point x
    :param width: float
         sigma of the Gaussian
    :param center: 3x1 numpy array
         position vector of a point at which the Gaussian is centerred
    :param structure: pylada Structure object
         used to account for the periodicity of the 
         center (basically the neighbouring centers)
    """

    centers = [center + dot(structure.cell,array([i,j,k])) \
                   for i in arange(-1,2) \
                   for j in arange(-1,2) \
                   for k in arange(-1,2)]

    return sum(array([exp(-dot(x-c,x-c)/width**2) for c in centers]))

########################################

def RSL_gen(atom_types=None,           \
            number_of_atoms=None,      \
            anions=None,               \
            angles_between=[60.,140.], \
            min_distance=1.7,          \
            atomic_scale=2.,           \
            cubic=False):
    """
    Generator of random supperlattice (RSL) structures. 

    For details please refer to:
    V. Stevanovic, Phys. Rev. Lett. 116, 075503 (2016)
    DOI:http://dx.doi.org/10.1103/PhysRevLett.116.075503

    The goal is to generate random structures that 
    favor cation-anion coordination. This is achieved
    by distributing different kinds of atoms randomly
    over two interpenetrating grids of points. The 
    grids are constructed using planes of a supperlattice
    defined by a randomly chosen reciprocal lattice 
    vector.

    :param atom_types: nx1 string list
         Chemical symbols of the atoms involved
    :param number_of_atoms: nx1 int list 
         Number of atoms of each type
    :param anions: nx1 0 and 1 list
         0-cations, 1-anions
    :param angles_between: 2x1 float list
         Range of angles between lattice vectors in degrees (not radians)
    :param min_distance: real
         Minimal distance bwtween two atoms
    :param atomic_scale: real
         Factors the minimal grid distance to determine the ionic size 
         Be aware this has nothing to do with real atomic/ionic sizes
         Change it if the atoms appear to be too close
    :param cubic: logical
         If true a cubic cell is constructed

    Example:
         This will generate a RSL structure 
         for MgO with 10 atoms in the unit cell

         RSL_gen(atom_types=['Mg','O'], 
                 number_of_atoms=[5,5], 
                 anions=[0,1], 
                 angles_between=[60.,140.],
                 min_distance=1.7,
                 atomic_scale=2.,
                 cubic=False)
    """

    # Attention!!! Throughout the code the 'hlp' and 'pom' are used as dummy variables
    # 'hlp' is short for 'help' and 'pom' is short for 'pomoc' (help in Serbian)

    # Setting the random cell 
    # a,b,c are the lengths of the lattice vectors
    # alpha, beta, gamma are the angles betweeen them 
    # the standard convetnion is respected (see http://en.wikipedia.org/wiki/Crystal_structure)
    # lengths are between 0.6 and 1.4 in units of scale
    # angles in the angles_between range

    s=Structure() # pylada structure object
    s.scale = 1.

    cell = zeros([3,3])

    if cubic:
            a = b = c = 1.
            alpha = beta = gamma = pi/2.

    else:
        a = 0.8*random()+0.6 
        b = 0.8*random()+0.6
        c = 0.8*random()+0.6
        
        alpha = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.
        beta  = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.
        gamma = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.

    a1 = a*array([1.,0.,0.]) # a1 is always along x
    a2 = b*array([cos(gamma),sin(gamma),0.]) # a2 is in the xy plane
        
    c1 = c*cos(beta) # projection onto x
    c2 = c/sin(gamma)*(-cos(beta)*cos(gamma) + cos(alpha)) # projection onto y

    ## Depending on the random choices projection onto z might turn negative
    ## repeating the process as long that's the case
    while c**2-(c1**2+c2**2)<=0.:
        a = 0.8*random()+0.6 
        b = 0.8*random()+0.6
        c = 0.8*random()+0.6
        
        alpha = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.
        beta  = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.
        gamma = ((angles_between[1]-angles_between[0])*random() + angles_between[0])*pi/180.

        a1 = a*array([1.,0.,0.])
        a2 = b*array([cos(gamma),sin(gamma),0.])
        
        c1 = c*cos(beta)
        c2 = c/sin(gamma)*(-cos(beta)*cos(gamma) + cos(alpha))
    ##############################################

    # third unit vector
    a3 = array([c1, c2, sqrt(c**2-(c1**2+c2**2))])
    
    # setting the unit cell matrix
    # unit vectors as rows in the matrix 
    cell = array([a1,a2,a3]) 

    # Have not gotten zero or negative volume yet, but asserting just in case
    assert (det(cell)>0.), "The cell volume is either zero or negative"

    # calculating the reciprocal cell
    rec_cell = 2*pi*transpose(inv(cell)) # reciprocal cell

    # pylada structure object takes unit vectors
    # as columns in the cell matrix
    # pay attention to the transpositions
    s.cell=transpose(cell) 

    # Setting two grids, one for the anions and one for the cations
    # This is done by taking the maxima and minima points of the COS(g*r) function 
    # where g is a random reciproical lattice vector (in other words a supperlattice)
    grid_an =[]
    grid_cat=[]

    # setting the g
    # chioce between 4 and 7 is to have large enough number of points, but not too dense grids
    # resulted from direct inspection of different 
    # change the range if needed
    slat = [randint(4,7),randint(4,7),randint(4,7)]
    g = slat[0]*rec_cell[0]+slat[1]*rec_cell[1]+slat[2]*rec_cell[2]

    # setting the real-space grids in such a way that they contain all the mentioned maxima and minima points
    # this is done in crystal coordinates (all numbers go from 0 to 1)
    grid_x=[x for x in arange(0.,1.,1./(2*slat[0]))]
    grid_y=[x for x in arange(0.,1.,1./(2*slat[1]))]
    grid_z=[x for x in arange(0.,1.,1./(2*slat[2]))]

    # Filling the candidate points for anions (maxima) and cations (minima)
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            for k in range(len(grid_z)):
                hlp=dot(s.cell,array([grid_x[i],grid_y[j],grid_z[k]]))
                if cos(dot(g,hlp)) == 1.:
                    grid_an.append(hlp)
                elif cos(dot(g,hlp)) == -1.:
                    grid_cat.append(hlp)
    
    # deleting the dummy variable 
    # just making sure that when using hlp later 
    # I do not pick up any previous value
    del hlp

    # Setting some kind of ionic sizes which will be used in creating the probability distribution,
    # the width of a Gaussian centered on every occupied point, basically
    # minimal distance between two grid points multiplied by the atomic_scale from the input
    cat_radius = atomic_scale*min([sqrt(dot(grid_cat[0]-grid_cat[j],grid_cat[0]-grid_cat[j])) \
                                       for j in range(1,len(grid_cat))])

    an_radius  = atomic_scale*min([sqrt(dot(grid_an[0]-grid_an[j],grid_an[0]-grid_an[j])) \
                                       for j in range(1,len(grid_an))])

    # Setting the initial prob. distribution
    probability_an  = array([1. for i in range(len(grid_an))])
    probability_cat = array([1. for i in range(len(grid_cat))])

    # Which grid points are occupied
    anions_occupy   = []
    cations_occupy  = []

    # distribute anions
    if sum(array(anions))>0:
        for i in range(len(atom_types)):
            if anions[i]==1:
                for j in range(number_of_atoms[i]):

                    # Placing the first anion
                    if len(anions_occupy) == 0:

                        # Chose on point on the anion grid
                        pom = randint(0,len(grid_an)-1)
                        anions_occupy.append(pom)
                        
                        # Adding the first anion
                        s.add_atom(grid_an[pom][0],grid_an[pom][1],grid_an[pom][2],atom_types[i])
                        # Recomputing the new prob distribution for anions
                        probability_an  = probability_an - array([prob(x,an_radius,grid_an[pom],s) for x in grid_an]) 
                        # Recomputing the new prob distribution for cations due to anions in the cell
                        probability_cat = probability_cat - array([prob(x,an_radius,grid_an[pom],s) for x in grid_cat]) 

                    # Placing the rest of anions
                    else:
                        # Setting the probability cutoff
                        p_crit = 0.9 
                        # Finding points on the anion grid that have the desired probability
                        hlp=[k for k in range(len(grid_an)) if probability_an[k]>p_crit and k not in anions_occupy]
                        # In case no points are found reduce the p_crit until you find something
                        while len(hlp)==0:
                            p_crit = p_crit-0.05
                            hlp=[k for k in range(len(grid_an)) if probability_an[k]>p_crit and k not in anions_occupy]
                        print('Working on anions')
                        # Repeat the procedure as for the first anion
                        pom = choice(hlp)
                        anions_occupy.append(pom)
                        s.add_atom(grid_an[pom][0],grid_an[pom][1],grid_an[pom][2],atom_types[i])
                        probability_an  = probability_an - array([prob(x,an_radius,grid_an[pom],s) for x in grid_an])
                        probability_cat = probability_cat - array([prob(x,an_radius,grid_an[pom],s) for x in grid_cat])


    # Distribute cations
    # The same as for anions
    if sum(array(anions))<len(anions):
        for i in range(len(atom_types)):
            if anions[i]==0:
                for j in range(number_of_atoms[i]):
                    p_crit = 0.9
                    hlp=[k for k in range(len(grid_cat)) if probability_cat[k]>p_crit and k not in cations_occupy]
                    while len(hlp)==0:
                        p_crit = p_crit-0.05
                        hlp=[k for k in range(len(grid_cat)) if probability_cat[k]>p_crit and k not in cations_occupy]
                    print('Working on cations')
                    pom = choice(hlp)
                    cations_occupy.append(pom)
                    s.add_atom(grid_cat[pom][0],grid_cat[pom][1],grid_cat[pom][2],atom_types[i]) 
                    probability_cat = probability_cat - array([prob(x,cat_radius,grid_cat[pom],s) for x in grid_cat])


    # Setting the scale so that the min_distance is fulfilled
    distances=[]
    for atom in s:

        # using pylada neighbors function to account for periodic 
        # boundary conditions
        distances.append(neighbors(s,1,atom.pos,0.1)[0][-1])

    while float(s.scale)*min(distances) < min_distance:
        s.scale = float(s.scale) + 0.2
        distances=[]
        for atom in s:
            distances.append(neighbors(s,1,atom.pos,0.1)[0][-1])

    return s
