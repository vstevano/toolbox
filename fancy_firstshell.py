###############################
#
#  This file is written by Vladan Stevanovic
#
###############################

from pylada.crystal import neighbors
################################
    
def fancy_firstshell(strc=None, atom=None, howmany=12, tol=None, types=None):
    """
    A function that uses the original pylada
    neighbors function to compute the first shell 
    coordination and neighbors of atoms.
    It still uses the original neighbors function, 
    just makes sure the output fullfils additional criteria. 

    :param strc: pylada structure

    :param atom: pylada Atom object
                 the atom whose first shell is calculated

    :param howmany: integer (default value = 12)
                    how many neighbors to look for in the 
                    initial pylada.crystal.neighbors call.
                    Can be a number larger than 12 but 
                    slows down the code.

    :param tol: dict
                Python dictionary with atom types as keys
                and tolerance factors for neighbors of
                each atom type. Tolerance is used as a 
                fraction increase from the closest neighbor.
                For example tol={'Si':0.2} will include all
                atom around Si that are found at distances
                between the closest neighbor and the 
                1.2*distance_to_the_closest_ngh from 
                the central atom

    :param types: dict
                  This parameter allows to include only 
                  certin atom types as neighbors. As for
                  tol the keys are the central atom types and
                  the values are the lists of atom types
                  that are included as neighbors. Caution,
                  the counting of neighbors stops with the 
                  first neighbor whose type is not included 
                  in the list.
    """

    nghs=[]
    
    if types==None:
    
        # Loop over pylada neighbors (arbitrary small tolerance)
        for ng in neighbors(strc,howmany,atom.pos,0.1):
            # If empty just add ng
            if len(nghs)==0:
                nghs.append(ng)
                continue

            # Make sure the distance is within the tolerance
            elif nghs[0][-1]<=ng[-1]<=(1.+tol[atom.type])*nghs[0][-1]:
                nghs.append(ng)

    else:
        
        # Loop over pylada neighbors (arbitrary small tolerance)
        for ng in neighbors(strc,howmany,atom.pos,0.1):

            if ng[0].type in types[atom.type]:
                
                # If empty just add ng
                if len(nghs)==0:
                    nghs.append(ng)
                    continue

                # Make sure the distance is within the tolerance and the
                # types of neighbors are in the types list
                elif (nghs[0][-1]<=ng[-1]<=(1.+tol[atom.type])*nghs[0][-1]):
                    nghs.append(ng)

            else:
                break
        
        return nghs
###############
