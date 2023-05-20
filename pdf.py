###############################
#
#  This file is written by Vladan Stevanovic
#
###############################


######################################
def g_r(strc,dr=0.1,cutoff_r=10.):
    """
    Function that computes pair distribution function
    between 0. and cutoff_r with the grid on the distance 
    axis defined by dr.
    Returns a dictionary with keys labeling the types of atoms 
    for which g(r) is calculated (e.g. 'Si-Si','Si-O', etc.)
    and the values are corresponding g(r)
    """

    from copy import deepcopy
    import numpy as np
    from pylada.crystal import neighbors
    import itertools 
    
    # rescale the structure just in case

    strc.cell=strc.cell*float(strc.scale)
    for ii in range(len(strc)):
        strc[ii].pos=strc[ii].pos*float(strc.scale)
    strc.scale=1.

    # number density
    n_dense = len(strc)/float(strc.volume)

    # the distance axis used to calculate rdf
    rs = np.arange(0.,cutoff_r+dr,dr)

    atom_types=list(set([atom.type for atom in strc]))
    atom_types_indices={at:[ii for ii in range(len(strc)) if strc[ii].type==at] for at in atom_types}

    g_r_dict={}

    for type1,type2 in itertools.product(atom_types,atom_types):

        n_dense_type2 = len(atom_types_indices[type2])/float(strc.volume)

        # dummy g_r
        gr = np.zeros(len(rs))

        for atom_index in atom_types_indices[type1]:

            # how many neighbors should we expect within cutoff_r
            # well, the number density times the volume (4pi/3*cutoff_r**3)
            n_nghs=n_dense*4*np.pi/3*cutoff_r**3

            # increase this number by 30% for good behavior
            n_nghs=int(1.3*n_nghs)
            
            # get the neighbors
            nghs = neighbors(strc,n_nghs,strc[atom_index].pos,0.3)

            # Increasing n_nghs as long as there the neighbor at a maximal
            # distance is closer than cutoff_r
            while nghs[-1][-1]<cutoff_r:
                #print("DOING THE n_nghs because %s is smaller than %s" %(nghs[-1][-1],cutoff_r))
                n_nghs=int(1.1*n_nghs)
                nghs = neighbors(strc,n_nghs,strc[atom_index].pos,0.3)
                #print("NOW %s vs. %s" %(nghs[-1][-1],cutoff_r))
                
            nghs = [ngh for ngh in nghs if ngh[0].type==type2]

            # count neighbors into bins between r and r+dr
            no_nghs=[]
            ii=0
            for r in rs[1:]:
                hlp=0.
                for jj in range(ii,len(nghs)):
                    if r-0.5*dr <= nghs[jj][-1] < r+0.5*dr:
                        hlp=hlp+1
                    else:
                        ii=jj
                        break

                no_nghs.append(hlp / (4*np.pi*r**2*dr))

            no_nghs.insert(0,0.)

            gr = gr + np.array(no_nghs)

#        gr = gr/n_dense_type2/len(atom_types_indices[type1])
        gr = gr/n_dense/len(atom_types_indices[type1])

        g_r_dict[type1+'-'+type2]=np.array(list(zip(rs,gr)))

    return g_r_dict

######################################
def n_r(strc,dr=0.1,cutoff_r=10.):
    """
    Function that computes number of atoms at a distance r 
    between 0. and cutoff_r with the grid on the distance 
    axis defined by dr
    """

    from copy import deepcopy
    import numpy as np
    from pylada.crystal import neighbors

    # rescale the structure
    cell=deepcopy(strc.cell)
    scale=deepcopy(strc.scale)
    positions=deepcopy([atom.pos for atom in strc])

    strc.scale=1.
    strc.cell=cell*float(scale)
    for ii in range(len(strc)):
        strc[ii].pos=np.array(positions[ii])*float(scale)
    # DONE

    # the distance axis used to calculate rdf
    rs = np.arange(0.,cutoff_r+dr,dr)

    # dummy g_r
    nr = np.zeros(len(rs))

    for atom in strc:

        # get the neighbors
        n_nghs = 100
        nghs = neighbors(strc,n_nghs,atom.pos,0.3)

        # make sure that neighbors are further away than cutoff_r
        while nghs[-1][-1]<cutoff_r:
            n_nghs = n_nghs+5
            nghs = neighbors(strc,n_nghs,atom.pos,0.3)

        # count neighbors into bins between r and r+dr
        no_nghs = [ len([ ng for ng in nghs if ng[-1] < r+0.5*dr ])  for r in rs[1:]]
        no_nghs.insert(0,0.)

        nr = nr + np.array(no_nghs)

    nr = nr/len(strc)

    return zip(rs,nr)
