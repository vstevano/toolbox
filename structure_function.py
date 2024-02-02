
import numpy as np
from itertools import product, combinations_with_replacement
from copy import deepcopy
#############################

class Sq():
    """
    Python class to compute structure function
    (structure factor) of any input structure.
    Periodicity is assumed.

    :param strc:     pylada structure

    :param dq:       real>0
                     grid spacing on the q-grid

    :param q_max:    real>0
                     Max q value

    :param x_ray:    Bool
                     Whether to simulate X-ray

    :param neutrons: Bool
                     Whether to simulate neutrons

    :param ones:     Bool
                     Generates structure function
                     with all atom scattering parameters
                     equal to 1.

    Example:

    from structure_function import Sq
    from pylada.crystal import read

    s=read.poscar('POSCAR')
    sq=Sq(strc=s, dq=0.01, q_max=20.0, x_ray=True, neutrons=False, ones=False)
    data=sq.strc_func_tot()
    """
    
    def __init__(self, strc=None, dq=0.01, q_max=25., x_ray=False, neutrons=False, ones=False):

        assert [x_ray,neutrons,ones].count(True)==1,\
            "One and only one of [x_ray, neutrons, ones] has to be set to True"
        
        self.strc     = self.rescale(strc)
        self.dq       = dq
        self.q_max    = q_max
        self.x_ray    = x_ray
        self.neutrons = neutrons
        self.ones     = ones
        self.dir_cell = self.strc.cell.T
        self.rec_cell = 2*np.pi*np.transpose(np.linalg.inv(self.dir_cell))
        self.q_grid   = np.linspace(0.,self.q_max+self.dq,int(self.q_max/self.dq)+2)

        return

    ###

    def rescale(self,strc):
        """
        Just a function to set the scale of the structure to 1
        """
        
        ss=deepcopy(strc)

        ss.cell=float(ss.scale)*ss.cell

        for ii in range(len(ss)):
            ss[ii].pos=float(ss.scale)*ss[ii].pos

        ss.scale=1.

        return ss

    ###

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

    ###
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
    
    ###
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

    ####
    def strc_func_tot(self):
        """
        Function to compute the total structure function
        (structure factor).
        """
        
        clustered_Gs = self.clustered_Gs()
        pos=[atom.pos for atom in self.strc]

        S_q = []
        
        for ii,gg_cluster in enumerate(clustered_Gs):

            qq=self.q_grid[ii]

            # Remove the q=0 term
            if qq==0:
                S_q.append([qq,0.])
                continue
            ##
            
            ffs=np.array(self.form_factors(q=qq))
            av_ff2=np.sum(ffs**2)/len(ffs)
    
            hlp=[]

            # Sum all values for G vectors with norms between
            # q and q+dq
            for gg in gg_cluster:
                hlp2=ffs*np.array([np.exp(1j*np.dot(gg[0],pp)) for pp in pos])
                hlp.append(np.sum([x*y for x,y in product(hlp2,hlp2.conjugate())])/av_ff2/len(self.strc))
                
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

        return np.array(S_q).real

    ###
    def strc_func_part_AL(self):
        """
        Function to compute partial structure functions
        (structure factors) in the Aschroft-Langreth formulation.
        The total structure function is also calculated.
        """
        
        from toolbox.atomic_scattering_params import ff, bb

        clustered_Gs = self.clustered_Gs()
        atom_types=set([atom.type for atom in self.strc])

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

                    # do not count the q=0 term
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


        # compute total from partials
        S_q = np.zeros(len(clustered_Gs),dtype=float)

        for key in S_q_ab.keys():

            types=key.split('-')
            qs=S_q_ab[key][:,0]

            # geting the structure factors
            if self.x_ray:
                ffs0=np.array([ff(qq,types[0]) for qq in qs])
                ffs1=np.array([ff(qq,types[1]) for qq in qs])

                ffs2=np.array([[ff(qq,atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

            elif self.neutrons:
                ffs0=np.array([bb(types[0]) for qq in qs])
                ffs1=np.array([bb(types[1]) for qq in qs])

                ffs2=np.array([[bb(atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)
                
                
            elif self.ones:
                ffs0=np.array([1. for qq in qs])
                ffs1=np.array([1. for qq in qs])

                ffs2=np.array([[1. for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)
            
            # Concentrations
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # Structure function terms
            # double count cross-terms!
            if types[0]!=types[1]:
                S_q = S_q + 2 * np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab[key][:,1]/av_ff2
            else:
                S_q = S_q +     np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab[key][:,1]/av_ff2

        S_q_ab['total']=np.array(list(zip(qs,S_q)))
            
        return S_q_ab

    ###

    def strc_func_total_AL(self,S_q_ab_AL=None):
        """
        Just a separate function to  compute total structure function
        from partials in the Ashcroft-Langreth form
        """

        from toolbox.atomic_scattering_params import ff, bb

        clustered_Gs = self.clustered_Gs()
        atom_types=set([atom.type for atom in self.strc])
        
        keys=list(S_q_ab_AL.keys())
        if 'total' in keys:
            del keys[keys.index('total')]

        
        S_q = np.zeros(len(clustered_Gs),dtype=float)

        for key in keys:

            types=key.split('-')
            qs=S_q_ab_AL[key][:,0]

            # geting the structure factors
            if self.x_ray:
                ffs0=np.array([ff(qq,types[0]) for qq in qs])
                ffs1=np.array([ff(qq,types[1]) for qq in qs])

                ffs2=np.array([[ff(qq,atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)

            elif self.neutrons:
                ffs0=np.array([bb(types[0]) for qq in qs])
                ffs1=np.array([bb(types[1]) for qq in qs])

                ffs2=np.array([[bb(atom.type)**2 for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)
                
                
            elif self.ones:
                ffs0=np.array([1. for qq in qs])
                ffs1=np.array([1. for qq in qs])

                ffs2=np.array([[1. for atom in self.strc] for qq in qs])
                av_ff2=np.sum(ffs2,axis=1)/len(self.strc)
            
            # Concentrations
            conc0=[atom.type for atom in self.strc].count(types[0])/len(self.strc)
            conc1=[atom.type for atom in self.strc].count(types[1])/len(self.strc)

            # Structure function terms
            # double count cross-terms!
            if types[0]!=types[1]:
                S_q = S_q + 2 * np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab_AL[key][:,1]/av_ff2
            else:
                S_q = S_q +     np.sqrt(conc0*conc1) * ffs0 * ffs1 * S_q_ab_AL[key][:,1]/av_ff2

        return np.array(list(zip(qs,S_q)))

    ######

