
from pylada.crystal import supercell, read, neighbors
import numpy as np
from pylada.ewald import ewald
from quantities import eV
################################

class CutSlab():
    """
    A class constructed to cut slabs
    of a given thickness from a given
    bulk structure for an arbitrary 
    set of Miller indices hkl. There are 
    also functions to produce termination 
    that minimizes the umber of broken 
    bonds of surface atoms and to compute 
    the dipole moment in the hkl direction
    from the atomic positions and the 
    oxidation states as charges

    Returns a slab from the 3D structure                                                              

    Takes a structure and makes a slab defined by the miller indices 
    with nlayers number of layers and vacuum defining the size of the 
    vacuum thickness. Variable cutoff determines the number of loops used 
    to get the direct lattice vectors perpendicular and parallel to 
    miller. For high index surfaces use larger cutoff value
    Warning: (1) cell is always set such that miller is alogn z-axes
             (2) nlayers and vacuum are always along z-axes.
    
    :param bulk:     pylada structure

    :param miller:   3x1 float64 array 
                     Miller indices defining the slab

    :param charges:  dict
                     'atom.type:charge' dictionary for 
                     calculations of the diole moment

    :param chem_pot: dict (default all species -1.)
                     chemical potentials of the species
                     needed for the energy minimization
                     instead of number of broken bonds

    :param min_thickness:  float (default 10.)
                           Minimal thickness of the slab in AA

    :param vacuum:   real (default 15.)
                     Vacuum thicness in angstroms

    :param cutoff:   integer (default 5)
                     Cutoff for finding the cell vectors of the slab structure
                     orthogonal and parrallel to hkl

    :param tol:      float (deafault 0.2) or dict={atom.type:tol for atom in bulk}
                     Tolerance for finding neighbors in fraction of the distance
                     to the closest neighbor, default is 20%. If dictionary
                     then each atom type has its own tolerance. This offers some 
                     flexibility when dealing with multinary compounds

    :param depth:    float (default 5. AA)
                     distance in AA from each termination 
                     plane in z-direction (hkl) withing 
                     which the undercoordinated atoms are 
                     looked for. For thin slabs (small nlayers 
                     of slanted a3 vector) depth is reset so it 
                     does not exceed the thicnkess of one layer.

    :param ewald_cutoff: float (default 100. Ry)
                         plane wavecutoff for the Ewald summation

    Example:

    from pylada.crystal import read
    from cut_slab import CutSlab
    import numpy as np
    ##############################

    bulk=read.poscar('POSCAR')
    miller=np.array([-2., 0., 1.])
    charges={'Ga':3., 'O':-2}
    min_thickness=20.
    vacuum=15.

    cs=CutSlab(bulk=bulk, miller=miller, charges=charges, min_thickness=min_thickness, vacuum=vacuum)
    slab=cs.make_slab()
    slab=cs.minimize_broken_bonds(slab=slab,verbose=True,maxiter=25)
    """
    
    ##########
    def __init__(self,\
                 bulk=None,\
                 miller=None,\
                 charges=None,\
                 chem_pot=None,\
                 #nlayers=5,\
                 min_thickness=10.,\
                 vacuum=15.,\
                 cutoff=5,\
                 tol=0.2,\
                 depth=5.,\
                 ewald_cutoff=100.):

        # Inputs
        self.bulk=bulk
        self.miller=miller
        self.charges=charges
        #self.nlayers=nlayers
        self.vacuum=vacuum
        self.cutoff=cutoff
        self.ewald_cutoff=ewald_cutoff
        self.min_thickness=min_thickness
        
        # Setting the tolerance
        if type(tol)==float: 
            self.tol={ttype:tol for ttype in set([atom.type for atom in bulk])}
        elif type(tol)==dict:
            self.tol=tol
        
	# Setting the depth 
        self.depth=depth
        
        # Derived inputs
        self.direct_cell=np.transpose(self.bulk.cell)
        self.reciprocal_cell=2*np.pi*np.transpose(np.linalg.inv(self.direct_cell))

        # Computing the neighbors (first shell) for the bulk atoms
        nghs=[]
        for atom in self.bulk:
            nghs.append(self.myneighbors(strc=self.bulk, atom=atom, howmany=12))

        self.bulk_nghs=nghs

        # Getting the coordination numbers
        coordinations=[len(ng) for ng in self.bulk_nghs]

        self.bulk_coordinations=coordinations
        
        # Computing the bond energies from the chemical potentials
        if chem_pot==None:
            self.chem_pot={ttype:-1. for ttype in set([atom.type for atom in self.bulk])}
        else:
            self.chem_pot=chem_pot
            
        bond_energies=[self.chem_pot[self.bulk[ii].type] / len(self.bulk_nghs[ii])\
                       for ii in range(len(self.bulk))]

        self.bond_energies=bond_energies

        # Computing the Madelung energy of the bulk (in eV)
        for ii in range(len(self.bulk)):
            self.bulk[ii].charge=charges[self.bulk[ii].type]
        
        madelung_energy=ewald(self.bulk, cutoff=self.ewald_cutoff).energy
        self.bulk_madelung_energy=float(madelung_energy.rescale(eV))
        
        # Printing a warning in case of multinary (ternary, quaternary,...) compounds
        if len(set([atom.type for atom in bulk]))>2:
            print("")
            print("WARNING: This is a multinary (ternary, quaternary,...) compound.")
            print("         Before proceeding check whether the default settings for the myneighbors")
            print("         function produce expected atom coordinations (investigate the bulk_nghs attribute).")
            print("         If not, adjust the 'tol' parameter(s) until you get it right. ")
            print("")
            
        return
    ##########
    
    def myneighbors(self, strc=None, atom=None, howmany=12):
        """
        A function that replaces the original pylada
        neighbors function that seems to be broken.
        It still uses the original one, just makes sure
        the output is correct. Also, it returns only the 
        first coordination shell.

        The tolerance is in fraction of the distance to 
        the closest atom and is 20% by default (test first)
        """

        nghs=[]

        # Loop over pylada neighbors (arbitrary small tolerance)
        for ng in neighbors(strc,howmany,atom.pos,0.1):
            # If empty just add ng
            if len(nghs)==0:
                nghs.append(ng)
                continue

            # Make sure the distance is within
            # the tolerance
            elif nghs[0][-1]<=ng[-1]<=(1.+self.tol[atom.type])*nghs[0][-1]:
#            elif nghs[0][-1]<=ng[-1]<=self.tol + nghs[0][-1]:
                nghs.append(ng)

        return nghs
    ##########

    def myneighbors_slab(self, slab=None, atom=None, howmany=12):
        """
        A function that replaces the original pylada
        neighbors function for the slab atoms.
        It uses the myneighbors function, just takes some care 
        that the distances to the first shell atoms do not 
        exceed those of the bulk.

        The tolerance is in fraction of the distance to 
        the closest atom and is the same as for the myneighbors.
        """

        nghs=[]

        # Loop ove pylada neighbors (arbitrary small tolerance)
        for ng in self.myneighbors(strc=slab, atom=atom, howmany=howmany):

            # Make sure the distances are all smaller or equal 
            # to those of the equivalent bulk atom

            # index of the equivalent bulk atom in the
            # bulk structure

            assert hasattr(atom,'site'),\
                "Something is off this atom does not have the site attribute! \
                Make sure to create the slabwith the make_slab() function."
            
            eq_site=atom.site

            # max distance to the atoms in the first shell
            # for the equivalent atom
            max_dist=max([bulk_ng[-1] for bulk_ng in self.bulk_nghs[eq_site]])

            # checking the distances and adding 0.01 AA just so to
            # avoid problems with numerical innacurracies
            if ng[-1]<=max_dist + 0.01:
                nghs.append(ng)

        return nghs
    ##########

    def make_slab(self):
        """
        A function that makes a slab of a given thickness
        defined by the hkl. It finds a unit cell with a1 
        and a2 orthogonal to hkl and a3 as parallel to hkl
        as possible. Then multiplies a3 with the nlayers 
        and adds vacuum.
        """

        from itertools import product
        
        orthogonal=[]  # list of lattice vectors orthogonal to miller
        
        for nn in product(np.linspace(self.cutoff,-self.cutoff, 2*self.cutoff+1),repeat=3):
            if np.dot(nn,self.miller) == 0 and np.linalg.norm(nn) > 1e-3:
                orthogonal.append(np.array(nn))

        # chose the shortest orthogonal and set it to be a1 lattice vector
        norm_orthogonal=[np.linalg.norm(np.dot(x,self.direct_cell)) for x in orthogonal]
        a1=orthogonal[norm_orthogonal.index(min(norm_orthogonal))]
        
        
        # chose the shortest orthogonal and not colinear with a1 and set it as a2
        in_plane=[]

        for x in orthogonal:
            if np.linalg.norm(x) > 1e-3:
                v=np.cross(np.dot(x,self.direct_cell),np.dot(a1,self.direct_cell))
                if np.linalg.norm(v) > 1e-3:
                    in_plane.append(x)

        norm_in_plane = [np.linalg.norm(np.dot(x,self.direct_cell)) for x in in_plane]
        a2 = in_plane[norm_in_plane.index(min(norm_in_plane))]

        a1 = np.dot(a1,self.direct_cell)
        a2 = np.dot(a2,self.direct_cell)

        # new cartesian axes z-along miller, x-along a1, and y-to define the right-hand orientation
        e1 = a1/np.linalg.norm(a1)
        e2 = a2 - np.dot(e1,a2)*e1
        e2 = e2/np.linalg.norm(e2)
        e3 = np.cross(e1,e2)

        # find vectors parallel to miller and set the shortest to be a3
        parallel = []

        for nn in product(np.linspace(self.cutoff,-self.cutoff, 2*self.cutoff+1),repeat=3):
            pom = np.dot(np.array(nn),self.direct_cell)
            if (np.linalg.norm(pom)-np.dot(e3,pom))<1e-8 and np.linalg.norm(pom)>1e-3:
                parallel.append(pom)

        # if there are no lattice vectors parallel to miller 
        if len(parallel)==0:
            for nn in product(np.linspace(self.cutoff,-self.cutoff, 2*self.cutoff+1),repeat=3):
                pom = np.dot(np.array(nn),self.direct_cell)
                if np.linalg.norm(pom)>1e-3 and np.dot(e3,pom)>1e-3:
                    parallel.append(pom)

        # For good behavior select only those that have nonzero orthogonal component
        parallel = [x for x in parallel if np.linalg.norm(x - np.dot(e1,x)*e1 - np.dot(e2,x)*e2)>1e-3 ]
        
        assert len(parallel)!=0,\
                "hkl=%s: Increase cutoff, found no lattice vectors parallel to (hkl)" %(self.miller)
        
        # Choosing a3 among those that have largest projection onto e3
        e3_proj = [np.dot(e3,pom)/np.linalg.norm(pom) for pom in parallel]
        max_e3_proj = max(e3_proj)
        parallel = [pom for ii,pom in enumerate(parallel) if np.abs(e3_proj[ii]-max_e3_proj)<0.1]
        norm_parallel = [np.linalg.norm(x) for x in parallel]

        assert len(norm_parallel)!=0,\
                 "hkl=%s: Increase cutoff, found no lattice vectors parallel to (hkl)" %(self.miller)


        # Now the shortest or of those
        a3 = parallel[norm_parallel.index(min(norm_parallel))]

        # How many time needs to be repeated to get above the min_thickness
        self.nlayers = int(np.ceil(self.min_thickness/np.dot(e3,a3)))

        # making a structure in the new unit cell - defined by the a1,a2,nlayers*a3 
        new_direct_cell = np.array([a1,a2,self.nlayers*a3])
        
        # make sure determinant is positive
        if np.linalg.det(new_direct_cell)<0.: new_direct_cell = array([-a1,a2,a3])

        assert np.linalg.det(new_direct_cell)>=0.9*np.linalg.det(self.direct_cell),\
            "hkl=%s Something is wrong your volume is equal to zero" %(self.miller)

        structure = supercell(self.bulk,np.transpose(new_direct_cell))

        # transformation matrix to new coordinates x' = dot(m,x)
        m = np.array([e1,e2,e3])

        # first the cell
        structure.cell=np.transpose(np.dot(new_direct_cell,np.transpose(m)))

        # resetting the depth so it does not exceed half a thincknes of the slab
        self.depth=min(self.depth,0.5*float(structure.cell[2][2]))
        
        # then the atoms
        for ii in range(len(structure)):
            structure[ii].pos=np.dot(m,structure[ii].pos)

        # checking whether there are atoms close to the cell faces and putting them back to zero
        for ii in range(len(structure)):
            scaled_pos = np.dot(structure[ii].pos,np.linalg.inv(np.transpose(structure.cell)))
            for jj in range(3):
                if abs(scaled_pos[jj]-1.)<1e-5:
                    scaled_pos[jj]=0.
            structure[ii].pos = np.dot(scaled_pos,np.transpose(structure.cell))

        # adding vaccum to the cell
        structure.cell = structure.cell + \
            np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,float(self.vacuum)/float(structure.scale)]])

        # translating atoms so that center of the slab and the center of the cell along z-axes coincide
        max_z = max([atom.pos[2] for atom in structure])
        min_z = min([atom.pos[2] for atom in structure])
        center_atoms = 0.5*(max_z+min_z)
        center_cell  = 0.5*structure.cell[2][2]

        for ii in range(len(structure)):
            structure[ii].pos = structure[ii].pos + np.array([0.,0.,center_cell-center_atoms])

        return structure
    ##########

    def get_under_coord(self, slab=None):
        """
        Returns indices and the coordinations of the 
        undercoordinated atoms in the slab relative 
        to the self.bulk

        IMPORTANT!!!
        Only the slab made from the bulk by the pylada.crystal.supercell
        function will work. This is because the supercell preserves 
        the relation of the supercell atoms to the sites they originated 
        from (through the atom.site attribute). 
        """

        # Check the coordination in the bulk first shell
        bulk_first_shell=[ [self.bulk[ii].type, len(self.bulk_nghs[ii])] for ii in range(len(self.bulk))]
        
        maxz=max([atom.pos[2] for atom in slab])
        minz=min([atom.pos[2] for atom in slab])

        # Find the undercoordinated atoms in the slab    
        under_coord=[]
    
        indices=[br for br in range(len(slab))\
                 if slab[br].pos[2]<=minz+self.depth/float(slab.scale)\
                 or\
                 maxz-self.depth/float(slab.scale)<=slab[br].pos[2]]
    
        for ii in indices:
            atom=slab[ii]

            # Index of the equivalent bulk atom to compare the coordination with
            eq_site=atom.site

            assert atom.type==self.bulk[eq_site].type,\
                "hkl=%s Atom %s in the slab does not correspond!!!" %(self.miller,ii)
        
            # Comparing coordination with the "equivalent" bulk atom.
            slab_ngh=self.myneighbors_slab(slab=slab, atom=atom)

            coordination=len(slab_ngh)
            
            if coordination != bulk_first_shell[eq_site][1]:
                under_coord.append([ii,\
                                    bulk_first_shell[eq_site][1]-coordination,\
                                    coordination,\
                                    bulk_first_shell[eq_site][1]]\
                                   )

        if len(under_coord)>0:
            assert all([x>0 for x in np.array(under_coord)[:,1]]),\
                "hkl=%s Something is off, coordination in the slab larger than in the bulk!!!" %(self.miller)

        # returns the list of undercoordinated atoms 
        # atom index in the slab, difference in coordination relative to the bulk, coordination, coordination in the bulk

        return under_coord
    ##########

    def get_top_under_coord(self, slab=None):
        """
        Returns indices and the coordinations of the 
        undercoordinated atoms at teh top surface of the slab relative 
        to their coordination in self.bulk

        IMPORTANT!!!
        Only the slab made from the bulk by the pylada.crystal.supercell
        function will work. This is because the supercell preserves 
        the relation of the supercell atoms to the sites they originated 
        from (through the atom.site attribute). 
        """

        # Check the coordination in the bulk first shell
        bulk_first_shell=[ [self.bulk[ii].type, len(self.bulk_nghs[ii])]\
                           for ii in range(len(self.bulk))]
        
        maxz=max([atom.pos[2] for atom in slab])

        # Find the undercoordinated atoms in the slab    
        under_coord=[]
    
        indices=[br for br in range(len(slab))\
                 if maxz-self.depth/float(slab.scale)<=slab[br].pos[2]]
    
        for ii in indices:
            atom=slab[ii]

            # Index of the equivalent bulk atom to compare the coordination with
            eq_site=atom.site

            assert atom.type==self.bulk[eq_site].type,\
                "hkl=%s Atom %s in the slab does not correspond!!!" %(self.miller,ii)
        
            # Comparing coordination with the "equivalent" bulk atom.
            slab_ngh=self.myneighbors_slab(slab=slab, atom=atom)

            coordination=len(slab_ngh)
            
            if coordination != bulk_first_shell[eq_site][1]:
                under_coord.append([ii,\
                                    bulk_first_shell[eq_site][1]-coordination,\
                                    coordination,\
                                    bulk_first_shell[eq_site][1]]\
                                   )

        if len(under_coord)>0:
            assert all([x>0 for x in np.array(under_coord)[:,1]]),\
                "hkl=%s Something is off, coordination in the slab larger than in the bulk!!!" %(self.miller)

        # returns the list of undercoordinated atoms 
        # atom index in the slab, difference in coordination relative to the bulk, coordination, coordination in the bulk

        return under_coord
    ##########
    
    def get_tot_broken_bonds(self, slab=None):
        """
        Function that returns the number of broken bonds
        """

        uc=self.get_under_coord(slab=slab)

        if len(uc)>0:
        #           Total                       Per undercoord atom
            return np.sum(np.array(uc)[:,1]), np.sum(np.array(uc)[:,1])/len(uc)
        elif len(uc)==0:
            return 0, 0.
    ##########

    def get_tot_top_broken_bonds(self, slab=None):
        """
        Function that returns the number of broken bonds
        """

        uc=self.get_top_under_coord(slab=slab)

        if len(uc)>0:
        #           Total                       Per undercoord atom
            return np.sum(np.array(uc)[:,1]), np.sum(np.array(uc)[:,1])/len(uc)
        elif len(uc)==0:
            return 0, 0.

    ##########
    
    def get_broken_bonds_per_area(self, slab=None):
        """
        Function that returns the number of broken bonds per area
        """

        uc=self.get_under_coord(slab=slab)
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))

        if len(uc)>0:        
            return np.sum(np.array(uc)[:,1])/2/area
        elif len(uc)==0:
            return 0.
    ##########

    def get_top_broken_bonds_per_area(self, slab=None):
        """
        Function that returns the number of broken bonds per area
        """

        uc=self.get_top_under_coord(slab=slab)
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))

        if len(uc)>0:
            return np.sum(np.array(uc)[:,1])/2/area
        elif len(uc)==0:
            return 0.
    ##########
    
    def get_bond_energy(self, slab=None):
        """
        Function that returns the bond energy 
        (energy increase due to bond breaking)
        for both sides of the slab.
        """

        und_coor=self.get_under_coord(slab=slab)

        bond_en=[]
        for uc in und_coor:
            atom_index=uc[0]
            
            eq_site=slab[atom_index].site
            b_e=np.round(-uc[1]*self.bond_energies[eq_site],8)
            
            bond_en.append([uc[0],b_e])
            
        return bond_en
    ##########

    def get_top_bond_energy(self, slab=None):
        """
        Function that returns the bond energy 
        (energy increase due to bond breaking)
        for the top termination.
        """

        und_coor=self.get_top_under_coord(slab=slab)
        
        bond_en=[]
        for uc in und_coor:
            atom_index=uc[0]
            
            eq_site=slab[atom_index].site
            b_e=np.round(-uc[1]*self.bond_energies[eq_site],6)

            bond_en.append([uc[0],b_e])
            
        return bond_en
    ##########

    def get_madelung_energy(self, slab=None):
        """
        Function that returns the Madelung energy of the slab in eV
        """

        for ii in range(len(slab)):
            slab[ii].charge=self.charges[slab[ii].type]
        
        madelung_energy=ewald(slab, cutoff=self.ewald_cutoff).energy
        
        return float(madelung_energy.rescale(eV))
    ##########
    
    def shift_to_bottom(self, slab=None, atom_index=None):
        """
        Function to shift an atom from the top to the bottom of the slab
        """
        shift_vector=np.transpose(slab.cell)[2] - np.array([0., 0., self.vacuum/float(slab.scale)])
        slab[atom_index].pos = slab[atom_index].pos - shift_vector

        return
    ##########
    
    def shift_to_top(self, slab=None, atom_index=None):
        """
        Function to shift an atom from the bottom to the top of the slab
        """
        shift_vector=np.transpose(slab.cell)[2] - array([0., 0., self.vacuum/float(slab.scale)])
        slab[atom_index].pos = slab[atom_index].pos + shift_vector

        return
    ##########
    
    def z_center(self, slab=None):
        return np.sum(np.array([atom.pos[2] for atom in slab]))/len(slab)
    ##########

    def get_dipole_moment(self, slab=None):
        """
        Calculates the dipole moment in the z-direction
        """

        z=[]
        ch=[]
        for atom in slab:
            z.append(atom.pos[2])
            ch.append(self.charges[atom.type])
            
        assert not abs(np.sum(np.array(ch)))>1e-5,\
            "hkl=%s System is not charge neutral!" %(self.miller)
        
        return np.round(np.sum(np.array(ch)*np.array(z)),4)
    ##########

    def re_center(self, slab=None):
        """
        Function to move the middle of the slab to the 
        middle of the cell (along z)
        """

        from copy import deepcopy

        pom=deepcopy(slab)
        
        max_z = max([atom.pos[2] for atom in pom])
        min_z = min([atom.pos[2] for atom in pom])
        center_atoms = 0.5*(max_z+min_z)
        center_cell  = 0.5*pom.cell[2][2]

        for ii in range(len(pom)):
            pom[ii].pos = pom[ii].pos + np.array([0.,0.,center_cell-center_atoms])

        return pom
    ##########

    def straighten_cell(self, slab=None):
        """
        Function to straighten the a3 unit vector along z.
        This can be done once the whole cutting business is 
        finished because the translational symmetry in the a3
        direction is broken anyhow.

        WARNING!!!
        This function uses the pylada.crystal.supercell function.
        Once applied the atom.site attributes wil no longer
        be derived from the bulk sites. Do this at hte very end
        not before.
        """

        cell=np.transpose(slab.cell)
        cell[2][0]=0.
        cell[2][1]=0.
        slab.cell=np.transpose(cell)
        r=np.diag([1,1,1])
        slab=supercell(slab,np.dot(slab.cell,r))

        return
    ##########
        
    def move_top2bottom(self, slab=None):
        """
        This function moves the topmost undercoordinated atoms
        to the bottom of the slab
        """
    
        from copy import deepcopy

        # Make a copy of the slab
        pom=deepcopy(slab)
        # Find the z-center so to know which of the undercoordinated atoms
        # are at the top surface
        rc=self.z_center(slab=pom)
        # Get the undercoordinated
        under_coord=self.get_top_under_coord(slab=pom)

        if len(under_coord)==0:
            return pom

        # Now, we want to select the most undercoordinated of the top undercoor to move.
        # This is done by selecting those whose undercoordination relative to
        # their coordination in the bulk is the largest. The criterion is:
        # vladan top_lowest_coord=max([uc[1]/uc[3] for uc in under_coord])
        top_lowest_coord=max([uc[1] for uc in under_coord])
        max_z_coord=max([pom[uc[0]].pos[2] for uc in under_coord if uc[1]==top_lowest_coord])
        
        # Now select and move
        if len(under_coord)>0:
            for uc in under_coord:
                #vladan if uc[1]/uc[3]==top_lowest_coord:
                #if uc[1]==top_lowest_coord:
                if uc[1]==top_lowest_coord and np.abs(max_z_coord-pom[uc[0]].pos[2])<0.1:
                    self.shift_to_bottom(slab=pom, atom_index=uc[0])
                    
        return pom
    ##########

    def minimize_broken_bonds(self, slab=None, verbose=True, maxiter=25):
        """
        A function to construct terminations 
        that minimize the number of broken bonds. 
        The idea is to move_top2bottom a numer of 
        times and then pick the slab with the minimal
        number of broken bonds. This might be an overkill
        but insures that the global minimum is found.
        The move_top2bottom is applied until 
        an entire 'layer' (see nlayers) is shifted.
        Finally, is two situations have the same
        minimial number of broken bonds the one with
        lower dipole moment is selcted

        :param slab: pylada structure object (slab)

        :param verbose: logical (to write messages or not)
                        does not work well in python.multiprocessing

        :param maxiter: int (default=25)
                        hard stop after maxiter iterations
        """
        
        # Get the relevant info
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))
        zcoord=[atom.pos[2] for atom in slab]
        zcenter=np.sum(zcoord)/len(slab)
        thickness=max(zcoord)-min(zcoord)
        no_broken=self.get_tot_broken_bonds(slab=slab)

        if no_broken[0]==0:
            if verbose:
                print("No of broken bonds = 0, nothing to do!")
            self.min_no_broken=0.
            self.min_no_broken_per_area=0.
            self.min_no_broken_per_atom=[0.]
            self.min_dipoles=[float(round(self.get_dipole_moment(slab=slab),4))]#/in_plane_area/thickness                
            return [slab]
        
        dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
        data = [[slab,zcenter,no_broken[0],no_broken[1],no_broken[0]/2/area,dipole]]
        
        if verbose:
            print("Start: no_broken={:3d},  dipole={: 10.3f}".format(no_broken[0],dipole))
            print("Shifting top2bottom until center shifts by {: 5.3f} angstroms or {:3d} iterations"\
                  .format(thickness/self.nlayers,maxiter))
            print(" ")


        # Now iterate until the entire layer is shifted to the bottom
        Iter=0

        if self.nlayers==1:
            nnll=1
        else:
            nnll=self.nlayers-1

        while (np.abs(data[-1][1]-data[0][1])<=thickness/nnll) and (Iter<maxiter):

            Iter=Iter+1

            # Move the top undercoordinated to the bottom
            slab=self.move_top2bottom(slab=slab)

            # Get the relevant info
            zcoord=[atom.pos[2] for atom in slab]
            zcenter=np.sum(zcoord)/len(slab)
            no_broken=self.get_tot_broken_bonds(slab=slab)

            if no_broken[0]==0:
                if verbose:
                    print("No of broken bonds = 0, stopping minimization!")
                dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
                data.append([slab,zcenter,0,0.,0.,dipole])
                break

            dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
            
            data.append([slab,zcenter,no_broken[0],no_broken[1],no_broken[0]/2/area,dipole])

            if verbose:
                print("Iter {:2d}: no_broken={:3d},  dipole={: 10.3f}, shift={: 8.3f}".\
                      format(Iter,no_broken[0],dipole,data[-1][1]-data[-2][1]))

        # Output after iterating
        no_broken_all=[float(np.round(dd[2],4)) for dd in data]
        min_no_broken_all=min(no_broken_all)

        min_indices=[]
        min_no_broken_per_atom=[]
        min_dipoles=[]

        for ii,xx in enumerate(no_broken_all):
            if not xx==min_no_broken_all:
                continue
            if (not float(round(data[ii][3],4)) in min_no_broken_per_atom) \
            or (not float(round(data[ii][5],4)) in min_dipoles):
                min_indices.append(ii)
                min_no_broken_per_atom.append(float(round(data[ii][3],4)))
                min_dipoles.append(float(round(data[ii][5],4)))
        
        out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices]

        self.min_no_broken=int(min_no_broken_all)
        self.min_no_broken_per_atom=min_no_broken_per_atom
        self.min_no_broken_per_area=float(round(min_no_broken_all/2/area,4)) 
        self.min_dipoles=min_dipoles
        
        return out_slabs
    ##########

    def minimize_top_broken_bonds(self, slab=None, verbose=True, maxiter=25):
        """
        A function to construct terminations 
        that minimize the number of broken bonds. 
        The idea is to move_top2bottom a numer of 
        times and then pick the slab with the minimal
        number of broken bonds. This might be an overkill
        but insures that the global minimum is found.
        The move_top2bottom is applied until 
        an entire 'layer' (see nlayers) is shifted.
        Finally, is two situations have the same
        minimial number of broken bonds the one with
        lower dipole moment is selcted

        :param slab: pylada structure object (slab)

        :param verbose: logical (to write messages or not)
                        does not work well in python.multiprocessing

        :param maxiter: int (default=25)
                        hard stop after maxiter iterations
        """
        
        # Get the relevant info
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))
        zcoord=[atom.pos[2] for atom in slab]
        zcenter=np.sum(zcoord)/len(slab)
        thickness=max(zcoord)-min(zcoord)
        top_broken=self.get_top_under_coord(slab=slab)

        if len(top_broken)==0:
            if verbose:
                print("No of top broken bonds = 0, nothing to do!")
            self.min_no_broken=0.
            self.min_no_broken_per_area=0.
            self.min_no_broken_per_atom=[0.]
            self.min_dipole=[float(round(self.get_dipole_moment(slab=slab),4))]#/in_plane_area/thickness
            return [slab]

        no_broken=np.sum(np.array(top_broken)[:,1])
        no_broken_per_atom=np.sum(np.array(top_broken)[:,1])/len(top_broken)
        dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
        data = [[slab,\
                 zcenter,\
                 top_broken,\
                 no_broken,\
                 no_broken_per_atom,\
                 no_broken/area,\
                 dipole]]
        
        if verbose:
            print("Start: no_broken={:3d},  dipole={: 10.3f}".format(no_broken,dipole))
            print("Shifting top2bottom until center shifts by {: 5.3f} angstroms or {:3d} iterations"\
                  .format(thickness/self.nlayers,maxiter))
            print(" ")


        # Now iterate until the entire layer is shifted to the bottom
        Iter=0

        if self.nlayers==1:
            nnll=1
        else:
            nnll=self.nlayers-1

        while (np.abs(data[-1][1]-data[0][1])<=thickness/nnll) and (Iter<maxiter):

            Iter=Iter+1

            # Move the top undercoordinated to the bottom
            slab=self.move_top2bottom(slab=slab)

            # Get the relevant info
            zcoord=[atom.pos[2] for atom in slab]
            zcenter=np.sum(zcoord)/len(slab)
            top_broken=self.get_top_under_coord(slab=slab)

            if len(top_broken)==0:
                if verbose:
                    print("No of top broken bonds = 0, stopping minimization!")

                no_broken=0.
                no_broken_per_atom=0.
                dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
                
                data.append([slab,\
                             zcenter,\
                             top_broken,\
                             no_broken,\
                             no_broken_per_atom,\
                             no_broken/area,\
                             dipole])
                                    
                break

            
            no_broken=np.sum(np.array(top_broken)[:,1])
            no_broken_per_atom=np.sum(np.array(top_broken)[:,1])/len(top_broken)
            dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
            
            data.append([slab,\
                         zcenter,\
                         top_broken,\
                         no_broken,\
                         no_broken_per_atom,\
                         no_broken/area,\
                         dipole])

            if verbose:
                print("Iter {:2d}: no_broken={:3d},  dipole={: 10.3f}, shift={: 8.3f}".\
                      format(Iter,no_broken,dipole,data[-1][1]-data[-2][1]))

        
        # Output after iterating
        no_broken_all=[dd[3] for dd in data]
        min_no_broken_all=min(no_broken_all)

        #min_indices=[ii for ii,xx in enumerate(no_broken_all) if xx==min_no_broken_all]
        min_indices=[]
        min_no_broken_per_atom=[]
        min_dipoles=[]

        for ii,xx in enumerate(no_broken_all):
            if not xx==min_no_broken_all:
                continue
            if (not float(round(data[ii][4],4)) in min_no_broken_per_atom) or \
                    (not float(round(data[ii][6],4)) in min_dipoles):
                min_indices.append(ii)
                min_no_broken_per_atom.append(float(round(data[ii][4],4)))
                min_dipoles.append(float(round(data[ii][6],4)))

        out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices]

        self.min_no_broken=int(min_no_broken_all)
        self.min_no_broken_per_atom=min_no_broken_per_atom
        self.min_no_broken_per_area=float(round(min_no_broken_all/area,4))
        self.min_dipoles=min_dipoles

        return out_slabs
    ##########

    def move_top2bottom_energy(self, slab=None):
        """
        This function moves the top, highest bond energy atoms
        to the bottom of the slab
        """
    
        from copy import deepcopy

        # Make a copy of the slab
        pom=deepcopy(slab)

        # Get the bond energies of top atoms
        top_bond_energies=self.get_top_bond_energy(slab=pom)

        if len(top_bond_energies)==0:
            return pom
        
        # Now, we want to select the highest bond energy of the top atoms to move.
        max_bond_energy=max(np.array(top_bond_energies)[:,1])
        max_z_coord=max([pom[tbe[0]].pos[2] for tbe in top_bond_energies if tbe[1]==max_bond_energy])
        
        # Now select and move
        for tbe in top_bond_energies:
            #if tbe[1]==max_bond_energy:
            if tbe[1]==max_bond_energy and np.abs(max_z_coord-pom[tbe[0]].pos[2])<0.1:
                self.shift_to_bottom(slab=pom, atom_index=tbe[0])
                    
        return pom
    ##########
    
    def minimize_bond_energy(self, slab=None, verbose=True, maxiter=25):
        """
        A function to construct terminations 
        that minimize the bond energy. 
        The idea is to move_top2bottom a numer of 
        times and then pick the slab with the minimal
        bond energy terminations. This might be an overkill
        but insures that the global minimum is found.
        The move_top2bottom is applied until 
        an entire 'layer' (see nlayers) is shifted.
        Finally, if two situations have the same
        minimial bond energy the one with
        lower dipole moment is selected

        :param slab: pylada structure object (slab)

        :param verbose: logical (to write messages or not)
                        does not work well in python.multiprocessing

        :param maxiter: int (default=25)
                        hard stop after maxiter iterations
        """
        
        # Get the relevant info
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))
        zcoord=[atom.pos[2] for atom in slab]
        zcenter=np.sum(zcoord)/len(slab)
        thickness=max(zcoord)-min(zcoord)
        bond_energy=self.get_bond_energy(slab=slab)

        if len(bond_energy)==0:
            if verbose:
                print("Bond energy = 0, nothing to do!")
            self.min_bond_energy=0.
            self.min_bond_energy_per_area=0.
            self.min_bond_energy_per_atom=[0.]
            self.min_dipole=[float(round(self.get_dipole_moment(slab=slab),4))]#/in_plane_area/thickness
            return [slab]
        
        tot_bond_energy=np.round(np.sum(np.array(bond_energy)[:,1]),6)
        bond_energy_per_atom=np.round(tot_bond_energy/len(bond_energy),6)
        dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
        data = [[slab,zcenter,tot_bond_energy,bond_energy_per_atom,tot_bond_energy/2./area,dipole]]
        
        if verbose:
            print("Start: bond_energy={: 10.3f},  dipole={: 10.3f}".format(tot_bond_energy,dipole))
            print("Shifting top2bottom until center shifts by {: 5.3f} angstroms or {:3d} iterations"\
                  .format(thickness/self.nlayers,maxiter))
            print(" ")


        # Now iterate until the entire layer is shifted to the bottom
        Iter=0

        if self.nlayers==1:
            nnll=1
        else:
            nnll=self.nlayers-1

        while (np.abs(data[-1][1]-data[0][1])<=thickness/nnll) and (Iter<maxiter):

            Iter=Iter+1

            # Move the top undercoordinated to the bottom
            slab=self.move_top2bottom_energy(slab=slab)

            # Get the relevant info
            zcoord=[atom.pos[2] for atom in slab]
            zcenter=np.sum(zcoord)/len(slab)
            bond_energy=self.get_bond_energy(slab=slab)

            if len(bond_energy)==0:
                if verbose:
                    print("Bond energy = 0, stopping minimization!")

                tot_bond_energy=0.
                bond_energy_per_atom=0.
                dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
                
                data.append([slab,zcenter,tot_bond_energy,bond_energy_per_atom,tot_bond_energy/2./area,dipole])
                    
                break


            tot_bond_energy=np.round(np.sum(np.array(bond_energy)[:,1]),6)
            bond_energy_per_atom=np.round(tot_bond_energy/len(bond_energy),6)
            dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
            data.append([slab,zcenter,tot_bond_energy,bond_energy_per_atom,tot_bond_energy/2./area,dipole])

            if verbose:
                print("Iter {:2d}: bond_energy={: 10.3f},  dipole={: 10.3f}, shift={: 8.3f}".\
                      format(Iter,tot_bond_energy,dipole,data[-1][1]-data[-2][1]))

        
        # Output after iterating
        bond_energy_all=[dd[2] for dd in data]
        min_bond_energy_all=min(bond_energy_all)
        
        min_indices=[]
        min_bond_energy_per_atom=[]
        min_dipoles=[]

        # find those that have bond energy equal to the minimum
        # but also differ in other quantities (indicating different
        # terminations) and output all of them
        for ii,xx in enumerate(bond_energy_all):
            if not xx==min_bond_energy_all:
                continue
            if (not float(round(data[ii][3],4)) in min_bond_energy_per_atom) or \
                    (not float(round(data[ii][5],4)) in min_dipoles):
                min_indices.append(ii)
                min_bond_energy_per_atom.append(float(round(data[ii][3],4)))
                min_dipoles.append(float(round(data[ii][5],4)))

        out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices]

        self.min_bond_energy=float(round(min_bond_energy_all,4))
        self.min_bond_energy_per_atom=min_bond_energy_per_atom
        self.min_bond_energy_per_area=float(round(min_bond_energy_all/2./area,4))
        self.min_dipoles=min_dipoles

        return out_slabs
    ##########

    def minimize_top_bond_energy(self, slab=None, verbose=True, maxiter=25):
        """
        A function to construct terminations 
        that minimize the bond energy. 
        The idea is to move_top2bottom a number of 
        times and then pick the slab with the minimal
        bond energy termination (top). This might be an overkill
        but insures that the global minimum is found.
        The move_top2bottom is applied until 
        an entire 'layer' (see nlayers) is shifted.
        Finally, if two situations have the same
        minimial bond energy the one with
        lower dipole moment is selcted

        :param slab: pylada structure object (slab)

        :param verbose: logical (to write messages or not)
                        does not work well in python.multiprocessing

        :param maxiter: int (default=25)
                        hard stop after maxiter iterations
        """
        
        # Get the relevant info
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))
        zcoord=[atom.pos[2] for atom in slab]
        zcenter=np.sum(zcoord)/len(slab)
        thickness=max(zcoord)-min(zcoord)
        top_bond_energy=self.get_top_bond_energy(slab=slab)

        if len(top_bond_energy)==0:
            if verbose:
                print("Top bond energy = 0, nothing to do!")
            self.min_bond_energy=0.
            self.min_bond_energy_per_area=0.
            self.min_bond_energy_per_atom=[0.]
            self.min_dipole=[float(round(self.get_dipole_moment(slab=slab),4))]#/in_plane_area/thickness
            return [slab]
        
        tot_top_bond_energy=np.round(np.sum(np.array(top_bond_energy)[:,1]),6)
        top_bond_energy_per_atom=np.round(tot_top_bond_energy/len(top_bond_energy),6)
        dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
        data = [[slab,\
                 zcenter,\
                 top_bond_energy,\
                 tot_top_bond_energy,\
                 top_bond_energy_per_atom,\
                 tot_top_bond_energy/area,\
                 dipole]]
        
        if verbose:
            print("Start: bond_energy={: 10.3f},  dipole={: 10.3f}".format(tot_top_bond_energy,dipole))
            print("Shifting top2bottom until center shifts by {: 5.3f} angstroms or {:3d} iterations"\
                  .format(thickness/self.nlayers,maxiter))
            print(" ")


        # Now iterate until the entire layer is shifted to the bottom
        Iter=0

        if self.nlayers==1:
            nnll=1
        else:
            nnll=self.nlayers-1

        while (np.abs(data[-1][1]-data[0][1])<=thickness/nnll) and (Iter<maxiter):

            Iter=Iter+1

            # Move the top undercoordinated to the bottom
            slab=self.move_top2bottom_energy(slab=slab)

            # Get the relevant info
            zcoord=[atom.pos[2] for atom in slab]
            zcenter=np.sum(zcoord)/len(slab)
            top_bond_energy=self.get_top_bond_energy(slab=slab)

            if len(top_bond_energy)==0:
                if verbose:
                    print("Top bond energy = 0, stopping minimization!")

                tot_top_bond_energy=0.
                top_bond_energy_per_atom=0.
                dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
                
                data.append([slab,\
                             zcenter,\
                             top_bond_energy,\
                             tot_top_bond_energy,\
                             top_bond_energy_per_atom,\
                             tot_top_bond_energy/area,\
                             dipole])
                
                break
            
            tot_top_bond_energy=np.round(np.sum(np.array(top_bond_energy)[:,1]),6)
            top_bond_energy_per_atom=np.round(tot_top_bond_energy/len(top_bond_energy),6)
            dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
            data.append([slab,\
                         zcenter,\
                         top_bond_energy,\
                         tot_top_bond_energy,\
                         top_bond_energy_per_atom,\
                         tot_top_bond_energy/area,\
                         dipole])

            if verbose:
                print("Iter {:2d}: bond_energy={: 10.3f},  dipole={: 10.3f}, shift={: 8.3f}".\
                      format(Iter,tot_top_bond_energy,dipole,data[-1][1]-data[-2][1]))

        # Output after iterating
        bond_energy_all=[dd[3] for dd in data]
        min_bond_energy_all=min(bond_energy_all)
  
        min_indices=[]
        min_bond_energy_per_atom=[]
        min_dipoles=[]

        # find those that have bond energy equal to the minimum
        # but also differ in other quantities (indicating different
        # terminations) and output all of them
        for ii,xx in enumerate(bond_energy_all):
            if not xx==min_bond_energy_all:
                continue
            if (not float(round(data[ii][4],4)) in min_bond_energy_per_atom) or \
                    (not float(round(data[ii][6],4)) in min_dipoles):
                min_indices.append(ii)
                min_bond_energy_per_atom.append(float(round(data[ii][4],4)))
                min_dipoles.append(float(round(data[ii][6],4)))

        out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices]

        self.min_bond_energy=float(round(min_bond_energy_all,4))
        self.min_bond_energy_per_atom=min_bond_energy_per_atom
        self.min_bond_energy_per_area=float(np.round(min_bond_energy_all/area,4))
        self.min_dipoles=min_dipoles

        return out_slabs
    ##########

    def minimize_madelung_energy(self, slab=None, verbose=True, maxiter=25):

        """
        A function to construct terminations 
        that minimize the Madelung energy. 
        The idea is to move_top2bottom a number of 
        times and then pick the slab with the minimal
        Madelung energy terminations. This might be an overkill
        but insures that the global minimum is found.
        The move_top2bottom is applied until 
        an entire 'layer' (see nlayers) is shifted.
        Finally, if two situations have the same
        minimial bond energy the one with
        lower dipole moment is selected

        :param slab: pylada structure object (slab)

        :param verbose: logical (to write messages or not)
                        does not work well in python.multiprocessing

        :param maxiter: int (default=25)
                        hard stop after maxiter iterations
        """
        
        # Get the relevant info
        area=np.linalg.norm(np.cross(slab.cell[:,0],slab.cell[:,1]))
        zcoord=[atom.pos[2] for atom in slab]
        zcenter=np.sum(zcoord)/len(slab)
        thickness=max(zcoord)-min(zcoord)
        madelung_energy=self.get_madelung_energy(slab=slab)
        madelung_energy_area=(self.get_madelung_energy(slab=slab)-self.bulk_madelung_energy*len(slab)/len(self.bulk))/area/2.
        dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
        data = [[slab,zcenter,madelung_energy,madelung_energy_area,dipole]]
        
        if verbose:
            print("Start: madelung_energy={: 10.3f},  dipole={: 10.3f}".format(madelung_energy,dipole))
            print("Shifting top2bottom until center shifts by {: 5.3f} angstroms or {:3d} iterations"\
                  .format(thickness/self.nlayers,maxiter))
            print(" ")


        # Now iterate until the entire layer is shifted to the bottom
        Iter=0

        if self.nlayers==1:
            nnll=1
        else:
            nnll=self.nlayers-1

        while (np.abs(data[-1][1]-data[0][1])<=thickness/nnll) and (Iter<maxiter):

            Iter=Iter+1

            # Move the top undercoordinated to the bottom
            #vladan slab=self.move_top2bottom_energy(slab=slab)
            slab=self.move_top2bottom(slab=slab)
            
            # Get the relevant info
            zcoord=[atom.pos[2] for atom in slab]
            zcenter=np.sum(zcoord)/len(slab)
            madelung_energy=self.get_madelung_energy(slab=slab)
            madelung_energy_area=(self.get_madelung_energy(slab=slab)-self.bulk_madelung_energy*len(slab)/len(self.bulk))/area/2.
            dipole=self.get_dipole_moment(slab=slab)#/in_plane_area/thickness
        
            data.append([slab,zcenter,madelung_energy,madelung_energy_area,dipole])

            if verbose:
                print("Iter {:2d}: madelung_energy={: 10.3f},  dipole={: 10.3f}, shift={: 8.3f}".\
                      format(Iter,madelung_energy,dipole,data[-1][1]-data[-2][1]))

        # Output after iterating
        madelung_energy_all=[dd[2] for dd in data]
        min_madelung_energy_all=round(min(madelung_energy_all),4)

        min_indices=[ii for ii,xx in enumerate(madelung_energy_all) if round(xx,4)==min_madelung_energy_all]

        # Seems that all slabs with the same Madelung en. and same dipole moment
        # are terminated the same way. Hence, outputing only one rather
        # than a bunch (lowest madelung).
        #out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices]
        out_slabs=[self.re_center(slab=data[mi][0]) for mi in min_indices][0]

        self.min_madelung_energy=min_madelung_energy_all
        self.min_madelung_energy_area=round(min_madelung_energy_all/2./area,4)
        self.min_dipoles=[float(round(data[mi][4],4)) for mi in min_indices]
        
        return out_slabs
    ##########
