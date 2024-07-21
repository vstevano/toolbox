################################################### 
#  This file is written Vladan Stevanovic
#
#  Copyright (C) 2024 Vladan Stevanovic
#  <http://www.gnu.org/licenses/>
#
###################################################
###################################################
#
# RECIPROCAL SPACE FUNCTIONS
#
###################################################
###################################################

import numpy as np
from itertools import product, combinations_with_replacement
from copy import deepcopy
#############################

class Sq():
    """
    Python class to compute various structure functions
    (structure factors) of any input structure.
    Periodicity is assumed.

    :param strc:      pylada structure

    :param dq:        real>0
                      grid spacing on the q-grid

    :param q_max:     real>0
                      Max q value

    :param x_ray:     Bool
                      Whether to simulate X-ray diffraction

    :param neutrons:  Bool
                      Whether to simulate neutron diffraction

    :param ones:      Bool
                      Generates structure function
                      with all atom scattering parameters
                      equal to 1. Pure Fourier transform
                      of the structure.
    
    Example:

    from structure_function import Sq
    from pylada.crystal import read

    s=read.poscar('POSCAR')
    sq=Sq(strc=s, dq=0.01, q_max=20.0, x_ray=True, neutrons=False, ones=False)
    sq.strc_func_part_AL()
    data=sq.part_sq_AL
    """

    ###############
    def __init__(self, strc=None, dq=0.01, q_max=25., x_ray=False, neutrons=False, ones=False):

        assert [x_ray,neutrons,ones].count(True)==1,\
            "One and only one of [x_ray, neutrons, ones] has to be set to True"

        self.strc       = self.rescale(strc)
        self.dq         = dq
        self.q_max      = q_max
        self.x_ray      = x_ray
        self.neutrons   = neutrons
        self.ones       = ones
        self.dir_cell   = self.strc.cell.T
        self.rec_cell   = 2*np.pi*np.transpose(np.linalg.inv(self.dir_cell))
        self.q_grid     = np.linspace(0.,self.q_max+self.dq,int(self.q_max/self.dq)+2)
        self.part_sq_AL = None
        self.part_sq_FZ = None
        self.tot_sq_AL  = None
        self.tot_sq_FZ  = None
        self.tot_sq_FZ_from_AL = None
        
        return

    ###############
    def rescale(self,strc):
        """
        A function to set the scale of the structure to 1
        so to not worry about that later.
        """
        
        ss=deepcopy(strc)

        ss.cell=float(ss.scale)*ss.cell

        for ii in range(len(ss)):
            ss[ii].pos=float(ss.scale)*ss[ii].pos

        ss.scale=1.

        return ss

    ###############
    def form_factors(self, q=None):
        """
        Gets the atomic form factors for
        all atoms in the input structure,
        and the norm of the q-vector = q.
        Both X_ray and neutron form factors
        are implemented. For the ones=True
        all form factors are set to one.

        :params q: Real
                   Norm of the q-vector
        """
        
        from toolbox.atomic_scattering_params import ff, bb
        
        if self.x_ray:
            fctrs=[ff(q,atom.type) for atom in self.strc]

        elif self.neutrons:
            fctrs=[bb(atom.type) for atom in self.strc]

        elif self.ones:
            fctrs=[1. for atom in self.strc]
            
        return fctrs

    ###############
    def get_Gs(self):
        """
        Calculate reciprocal lattice vectors
        inside the sphere with the radius q_max.
        A bit more, but that is for good behavior.
        """

        from operator import itemgetter

        n_max=max([self.q_max/np.linalg.norm(self.rec_cell[ii]) for ii in range(3)])
        n_max=int(round(n_max,0))+1
        
        Gs=[]
        
        for ns in product(np.linspace(-n_max,n_max,2*n_max+1),repeat=3):
            gg=np.dot(ns,self.rec_cell)
            Gs.append([gg,np.linalg.norm(gg)])

        return sorted(Gs,key=itemgetter(1))

    ###############
    def clustered_Gs(self):
        """
        Calculate Gs and cluster them into groups
        with similar norms.
        """
        
        Gs=self.get_Gs()

        clustered_Gs=[]
        ii=0

        for jj in range(len(self.q_grid)-1):
            hlp=[]
            for kk in range(ii,len(Gs)):
                gg=Gs[kk]
                if self.q_grid[jj]<=gg[1]<self.q_grid[jj+1]:
                    hlp.append(gg)
                else:
                    ii=deepcopy(kk)
                    break

            clustered_Gs.append(hlp)

        return clustered_Gs

    ###############
    def strc_func_part_AL(self):
        """
        Function to compute partial structure functions
        (structure factors) in the Aschroft-Langreth formulation.
        """
        
        clustered_Gs = self.clustered_Gs()
        atom_types=list(set([atom.type for atom in self.strc]))
        atom_types.sort()

        S_q_ab={}

        # compute pairwise partial S_q
        for type1,type2 in combinations_with_replacement(atom_types,2):

            pos1=[atom.pos for atom in self.strc if atom.type==type1]
            pos2=[atom.pos for atom in self.strc if atom.type==type2]

            S_q = []
        
            for ii,gg_cluster in enumerate(clustered_Gs):
                
                qq=self.q_grid[ii]
                
                hlp=[]
                
                # Sum all values for G vectors with norms between
                # q and q+dq
                for gg in gg_cluster:

                    # set the q=0 term to 0
                    if len(gg_cluster)==1 and np.linalg.norm(gg[0])==0:
                        hlp.append(0.)
                        continue
                    
                    sq=np.array([np.exp(-1j*np.dot(gg[0],pp1-pp2)) for pp1 in pos1 for pp2 in pos2])
                    hlp.append(np.sum(sq)/np.sqrt(len(pos1)*len(pos2)))

                if len(hlp)>0:
                    # Spherical average
                    # Only terms with non-zero contribution to S_q are included in the average
                    count_nonzeros=[np.linalg.norm(x) > 1e-8 for x in hlp].count(True)
                    if count_nonzeros!=0:
                        S_q.append([ qq,np.sum(hlp) / count_nonzeros] )
                    else:
                        S_q.append([qq,0.])

                else:
                    S_q.append([qq,0.])

            S_q_ab['%s-%s' %(type1,type2)]=np.array(S_q).real

        self.part_sq_AL=S_q_ab
        
        return 

    ###############
    def strc_func_tot_AL(self):
        """
        Function to compute the total AL structure function
        from the AL partials
        """

        from toolbox.atomic_scattering_params import ff, bb

        # Get the partials first
        if self.part_sq_AL == None:
            self.strc_func_part_AL()
            S_q_ab=self.part_sq_AL
        else:
            S_q_ab=self.part_sq_AL

        # miseleneous stuff
        clustered_Gs = self.clustered_Gs()
        atom_types=set([atom.type for atom in self.strc])
        
        # compute total from partials
        S_q = np.zeros(len(clustered_Gs),dtype=float)

        for key in S_q_ab.keys():

            types=key.split('-')
            qs=S_q_ab[key][:,0]

            # geting the atomic form factors
            if self.x_ray:
                ffs0=np.array([ff(qq,types[0]) for qq in qs])
                ffs1=np.array([ff(qq,types[1]) for qq in qs])

                ffs2=np.array([[ff(qq,atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                #ffs=np.array([[ff(qq,atom.type) for atom in self.strc] for qq in qs])
                #av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2

                
            elif self.neutrons:
                ffs0=np.array([bb(types[0]) for qq in qs])
                ffs1=np.array([bb(types[1]) for qq in qs])

                ffs2=np.array([[bb(atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                #ffs=np.array([[bb(atom.type) for atom in self.strc] for qq in qs])
                #av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2
                
                
            elif self.ones:
                ffs0=np.array([1. for qq in qs])
                ffs1=np.array([1. for qq in qs])

                ffs2=np.array([[1. for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                #av_ff_2=1.
            
            # Atomic fractions
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # Structure function terms
            # double count cross-terms!
            if types[0]!=types[1]:
                S_q = S_q + 2 * np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab[key][:,1]/av_ff2
            else:
                S_q = S_q +     np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab[key][:,1]/av_ff2
                #S_q = S_q +     np.sqrt(conc0*conc1) * ffs0 * ffs1/av_ff_2 * (S_q_ab[key][:,1]-1)

        self.tot_sq_AL = np.array(list(zip(qs,S_q)))

        return 

    ###############
    def strc_func_tot_FZ_from_AL(self):
        """
        Function to compute total Faber-Ziman structure function
        from partials in the Ashcroft-Langreth form
        """

        from toolbox.atomic_scattering_params import ff, bb

        if self.part_sq_AL == None:
            self.strc_func_part_AL()
            S_q_ab_AL=self.part_sq_AL
        else:
            S_q_ab_AL=self.part_sq_AL
        
        
        clustered_Gs = self.clustered_Gs()
        atom_types=set([atom.type for atom in self.strc])
        
        S_q = np.zeros(len(clustered_Gs),dtype=float)

        for key in S_q_ab_AL.keys():

            types=key.split('-')
            qs=S_q_ab_AL[key][:,0]

            # geting the structure factors
            if self.x_ray:
                ffs0=np.array([ff(qq,types[0]) for qq in qs])
                ffs1=np.array([ff(qq,types[1]) for qq in qs])

                #ffs2=np.array([[ff(qq,atom.type)**2 for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                ffs=np.array([[ff(qq,atom.type) for atom in self.strc] for qq in qs])
                av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2

                
            elif self.neutrons:
                ffs0=np.array([bb(types[0]) for qq in qs])
                ffs1=np.array([bb(types[1]) for qq in qs])

                #ffs2=np.array([[bb(atom.type)**2 for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                ffs=np.array([[bb(atom.type) for atom in self.strc] for qq in qs])
                av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2
                
                
            elif self.ones:
                ffs0=np.array([1. for qq in qs])
                ffs1=np.array([1. for qq in qs])

                #ffs2=np.array([[1. for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                av_ff_2=1.
            
            # Concentrations
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # Structure function terms
            # double count cross-terms!
            if types[0]!=types[1]:
                S_q = S_q + 2 * np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab_AL[key][:,1]/av_ff_2
            else:
                S_q = S_q +     np.sqrt(conc0*conc1) * ffs0 * ffs1/av_ff_2 * (S_q_ab_AL[key][:,1]-1.)

        self.tot_sq_FZ_from_AL=np.array(list(zip(qs,S_q + 1.)))

        return 

    ###############
    def strc_func_part_FZ(self):
        """
        Function to compute partial structure functions
        (structure factors) in the Faber-Ziman formulation.
        Uses Aschroft-Langreth partials.
        """
        
        clustered_Gs = self.clustered_Gs()
        atom_types   = set([atom.type for atom in self.strc])

        if self.part_sq_AL == None:
            self.strc_func_part_AL()
            S_q_ab_AL=self.part_sq_AL
        else:
            S_q_ab_AL=self.part_sq_AL

        S_q_ab_FZ={}
            
        for key,S_q in S_q_ab_AL.items():
            qs=S_q[:,0]
            ss_AL=S_q[:,1]

            types=key.split('-')
            
            # Concentrations
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            if types[0]!=types[1]:
                ss_FZ = 1./np.sqrt(conc0*conc1) * ss_AL + 1.
                
            elif types[0]==types[1]:
                ss_FZ = 1./np.sqrt(conc0*conc1) * ss_AL + 1. - 1./conc1

            S_q_ab_FZ[key]=np.array(list(zip(qs,np.array(ss_FZ).real)))
                
        self.part_sq_FZ = S_q_ab_FZ
        
        return 

    ###############
    def strc_func_tot_FZ(self):
        """
        Function to compute total Faber-Ziman structure function
        from Faber-Ziman partials
        """

        from toolbox.atomic_scattering_params import ff, bb

        if self.part_sq_FZ == None:
            self.strc_func_part_FZ()
            S_q_ab_FZ=self.part_sq_FZ
        else:
            S_q_ab_FZ=self.part_sq_FZ
        
        
        clustered_Gs = self.clustered_Gs()
        atom_types=set([atom.type for atom in self.strc])
        
        S_q = np.zeros(len(clustered_Gs),dtype=float)

        for key in S_q_ab_FZ.keys():

            types=key.split('-')
            qs=S_q_ab_FZ[key][:,0]

            # geting the structure factors
            if self.x_ray:
                ffs0=np.array([ff(qq,types[0]) for qq in qs])
                ffs1=np.array([ff(qq,types[1]) for qq in qs])

                #ffs2=np.array([[ff(qq,atom.type)**2 for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                ffs=np.array([[ff(qq,atom.type) for atom in self.strc] for qq in qs])
                av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2

                
            elif self.neutrons:
                ffs0=np.array([bb(types[0]) for qq in qs])
                ffs1=np.array([bb(types[1]) for qq in qs])

                #ffs2=np.array([[bb(atom.type)**2 for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                ffs=np.array([[bb(atom.type) for atom in self.strc] for qq in qs])
                av_ff_2=(np.sum(ffs,axis=1)/len(self.strc))**2
                
                
            elif self.ones:
                ffs0=np.array([1. for qq in qs])
                ffs1=np.array([1. for qq in qs])

                #ffs2=np.array([[1. for atom in self.strc] for qq in qs])
                #av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

                av_ff_2=1.
            
            # Concentrations
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # Structure function terms
            # double count cross-terms!
            if types[0]!=types[1]:
                S_q = S_q + 2 * conc0*conc1 * ffs0 * ffs1/av_ff_2 * S_q_ab_FZ[key][:,1]
            else:
                S_q = S_q +     conc0*conc1 * ffs0 * ffs1/av_ff_2 * S_q_ab_FZ[key][:,1]

        self.tot_sq_FZ=np.array(list(zip(qs,S_q)))

        return 


    ###############
    def part_PDFFT(self,rs=np.linspace(0.,15.,10001)):
        """
        A function to compute the Fourier transform of the partial S_q in the Aschroft-Langreth
        form to get the partial pair distribution function.

        :param rs: 1xN 
                   list of r values for which to do the transform

        """

        import numpy as np

        if self.part_sq_AL == None:
            self.strc_func_part_AL()
            S_q_ab_AL=self.part_sq_AL
        else:
            S_q_ab_AL=self.part_sq_AL
        
        #dq=S_q_ab_AL[:,0][1]-S_q_ab_AL[:,0][0]
        #q_max=max(S_q_ab_AL[:,0])

        
        ##
        def M(q):
            """
            A modification function to deal with the truncated F transform
            """

            return q_max/(q*np.pi) * np.sin(q*np.pi/q_max)
            #return 1.
        
        ##

        # the dicitionary of partial PDFs
        g_r={}

        for key,sq in S_q_ab_AL.items():

            # q grid
            qs=sq[:,0]
            q_max=max(qs)
            dq=qs[1]-qs[0]
            
            # Atom types
            types=key.split('-')

            # Atom fractions
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # create an empty array of the length 
            pdf=np.zeros(len(rs),dtype=float)

            # Doing the sine fourier transform for each r
            # Note that the modification function contains dicision by zero
            # which is not a problem as the q=0 value of the integrand is zero.
            # That is why qs[1:] * sq[:,1][1:] * ...
            #
            for ii,rr in enumerate(rs):
                if types[0]!=types[1]:
                    pdf[ii] = 2./np.pi * np.sqrt(conc0*conc1)**(0.5) * \
                        np.sum( qs[1:] * sq[:,1][1:] * np.sin(rr*qs[1:]) * M(qs[1:]))

                elif types[0]==types[1]:
                    pdf[ii] = 2./np.pi * np.sqrt(conc0*conc1)**(0.5) * \
                        np.sum( qs[1:] * (sq[:,1][1:]-1.) * np.sin(rr*qs[1:]) * M(qs[1:]))

            # This is the reduced PDF (D(r)).
            # To get the g_ab(r) = 1. + 1./(4*pi*n*r) * D_ab(r)
            # There is again division by 0 for r=0 that needs some care

            # The number density
            n_dens = len(self.strc)/float(self.strc.volume)

            # the pdf
            pdf[1:] = 1. + 1./(4*np.pi*n_dens*rs[1:])*pdf[1:]

            # Create an array that has rs and pdf as columns
            out_pdf=np.concatenate((rs,pdf),axis=0)
            out_pdf=np.reshape(out_pdf,(2,len(out_pdf)//2)).T

            # Add to the g_r dictionary
            g_r[key]=out_pdf

        return g_r
    ##############################
    ##############################

###################################################
###################################################
#
# REAL SPACE FUNCTIONS
#
###################################################
###################################################
#
class PDF():

    """
    Python class to compute the partial pair distribution functions of
    a given structure.

    :param strc:      pylada structure

    :param dr:        real>0
                      grid spacing on the r-grid

    :param r_max:     real>0
                      Max r value

    
    Example:

    from structure_functions import PDF
    from pylada.crystal import read

    s=read.poscar('POSCAR')
    pdf=PDF(strc=s, dr=0.1, r_max=20.0)
    gr=pdf.pdf()

    IMPORTANT!!!

    Because of boundary effects, compute the pdfs with r_max larger than you
    need. For example, set r_max=25-30. AA, to plot from 0. to 20. In other words,
    the result needs to be converged w.r.t. r_max.
    
    """

    def __init__(self, strc=None, dr=0.1, r_max=25.):

        from pylada.crystal import supercell

        # rescale the structure
        self.strc = self.rescale(strc)

        # calculate the total number density
        self.n_dense = len(self.strc)/float(self.strc.volume)

        # extract the types of atoms and sort in alphabetical order
        atom_types = list(set([atom.type for atom in self.strc]))
        atom_types.sort()
        self.atom_types = atom_types
        
        # get the indices of each atom type
        self.atom_types_indices = {at:[ii for ii in range(len(self.strc)) if self.strc[ii].type==at] for at in self.atom_types}

        # set the radial grid
        self.dr      = dr
        self.r_max   = r_max
        self.rs      = np.linspace(0.,r_max,int(r_max/dr)+1)

        # create the supercell containing all the needed atoms
        repeat = max([ r_max/np.linalg.norm(self.strc.cell[:,ii]) for ii in range(3)])
        repeat = 5*int(np.round(repeat,0))

        self.strc_sc = supercell(self.strc,np.dot(np.diag([repeat,repeat,repeat]),self.strc.cell))

        # Lattice vector to translate close to the geometric center of the supercell
        rc=0.5*np.sum(self.strc_sc.cell.T,0)
        ns=np.round(np.dot(rc,np.linalg.inv(self.strc.cell.T)),0)

        self.transl_r = np.dot(ns,self.strc.cell.T)
        
        return

    ###############

    def rescale(self,strc):
        """
        A function to set the scale of the structure to 1
        so to not worry about that later.
        """
        
        ss=deepcopy(strc)

        ss.cell=float(ss.scale)*ss.cell

        for ii in range(len(ss)):
            ss[ii].pos=float(ss.scale)*ss[ii].pos

        ss.scale=1.

        return ss

    ###############

    def pdf(self):
        """
        Function to compute the partial pair distribution functions (PDF)
        """

        g_r_dict={}

        for type1,type2 in product(self.atom_types,self.atom_types):

            n_dense_type2 = len(self.atom_types_indices[type2])/float(self.strc.volume)

            # dummy g_r
            gr = np.zeros(len(self.rs))

            # loop over the a atoms
            for atom_index in self.atom_types_indices[type1]:

                # position of the a atom shifted close to the center
                pos_a = self.strc[atom_index].pos + self.transl_r
                
                # get the distances to all b
                nghs=[np.linalg.norm(atom.pos - pos_a) \
                      for atom in self.strc_sc if atom.type==type2]

                # sort the distances in ascending order
                nghs.sort()

                # remove the distance to itself (0.)
                if nghs[0]<1e-6:
                    del nghs[0]
                
                # count how many neighbors in each bin between r and r+dr
                no_nghs=[]
                ii=0

                for r in self.rs[1:]:
                    hlp=0.

                    for jj in range(ii,len(nghs)):
                        if r-0.5*self.dr <= nghs[jj] < r+0.5*self.dr:
                            hlp=hlp+1
                        else:
                            ii=jj
                            break
                    
                    no_nghs.append(hlp / (4*np.pi*r**2*self.dr))

                no_nghs.insert(0,0.)

                gr = gr + np.array(no_nghs)

            # normalize appropriately (goes to 1. for large r)
            gr = gr/n_dense_type2/len(self.atom_types_indices[type1])

            # add to the dictionary
            g_r_dict[type1+'-'+type2]=np.array(list(zip(self.rs,gr)))

        return g_r_dict
    #######################
    #######################
