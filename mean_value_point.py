###############################
#  This code was written by Vladan Stevanovic
#
###############################

class MVP():
    """
    A code to compute the Mean Value (Baldereschi) Point in the first Brillouin zone.
    For a detailed explanation of the fundamentals of the mean valu point and how
    to compute it please check:

    A. Baldcreschi, 'Mean-Value Point in the Brillouin Zone', Phys. Rev. B 7, 5212 (1973)
    DOI: https://doi.org/10.1103/PhysRevB.7.5212
    
    This code follows directly the reasonong from Baldereschi, with the caveat that the
    system of equations that needs to be solved to obtain the mean value point is
    solved numerically on the grid of k-points. 
    
    IMPORTANT!!! Please understand that the code is solving the system of nonlinear equations. 
                 Also, the solution (including the existance of the solution) depends on the 
                 number of k-points. Please chack the convergence of te solution
                 w.r.t. the nkpts.

    :param strc: pylada strcutre pbject

    :param zero: float (default 1e-12)
                 Tolerance factor for what we call 
                 zero value of various functions

    :param gen_tol: float (default 1e-5)
                 Tolerance factor for differentiating
                 vectors and other quantities 

    :param nkpts: int (default 2e+7)
                 Total number of k-point in a grid.
                 This allows different discretization
                 of reciprocal lattice vectors to maintain
                 the same spacing.

    :param R_range: int (default 4)
                 Integer parameter that defines how many 
                 R_vectors are generated and subsequently used 
                 in R_stars. The total number of R_vectors
                 is (2*R_range+1)**3

    :calc_f1_zeros: bool (default True)
                 Whether to compute the zeros of f1.
                 The f1 is just a fully symmetrized wave
                 from the first R_star.

    :only_lowest_norm: bool (default True)
                 There will generally be multiple solutions
                 for the mean value point (likely symmetry equivalent).
                 If this paramter is True only the lowest norm 
                 solution is in the output. If False, all
                 are reported.

    EXAMPLE:

    >>> from mean_value_point import MVP
    >>> from pylada.crystal import read
    >>> s=read.poscar('POSCAR_bcc')
    >>> mvp=MVP(strc=s,nkpts=3e+5)
    >>> mvp_point=mvp()
    >>> mvp_point/(2*np.pi)
    ... array([0.5       , 0.16666667, 0.16666667])

    """

    def __init__(self,\
                 strc=None,\
                 zero=1e-12,\
                 gen_tol=1e-5,\
                 nkpts=2e+7,\
                 R_range=4,\
                 calc_f1_zeros=True,\
                 only_lowest_norm=True):

        import numpy as np
        import spglib
        from toolbox.format_spglib import to_spglib, from_spglib

        # Structure
        # rescaling just in case
        strc.cell=strc.cell*float(strc.scale)
        for ii in range(len(strc)):
            strc[ii].pos=strc[ii].pos*float(strc.scale)
        strc.scale=1.

        self.strc=strc

        # Tolerance factor for what we call zero value of various functions 
        self.zero=zero

        # Tolerance factor for differentiating vectors and other quantities
        self.gen_tol=gen_tol

        # Number of k-grid points
        self.nkpts=nkpts
        
        # np.linspace(-R_range,R_range,2*R_range+1) is the range of lattice vectors
        # that will form the stars used in creating the Gamma_1 IR
        self.R_range=R_range        

        # Whether to report only the lowest norm solution
        # or all of them.
        self.only_lowest_norm=only_lowest_norm
        
        # Direct cell and reciprocal cell (vectors are rows)
        self.cell=np.transpose(self.strc.cell)*float(self.strc.scale)
        self.rcell=2*np.pi*np.transpose(np.linalg.inv(self.cell))

        # Getting the symmetry operations
        # with generous tolerancies
        symmetries = spglib.get_symmetry(
            to_spglib(self.strc),
            symprec=1e-02,
            angle_tolerance=-1.0,
            is_magnetic=True,
        )

        # Extractuing the rotations only
        # this includes mirros, checked the determinents
        # and there are instances with det=-1.
        # I still do not know whether rotations include point
        # operations that part of the screw axes and/or glide planes?
        rotations = symmetries['rotations']

        # Converting rotations to Cartesian coordinates
        self.rotations = [np.dot(self.cell.T,np.dot(R_c,np.linalg.inv(self.cell.T))) for R_c in rotations]
        
        # Getting the stars in real space
        self.R_stars=self.get_R_stars()

        # Getting the k-point grid
        # Calculating actual k-points (as a list)
        self.k_points=self.get_kpts_grid()

        if calc_f1_zeros:

            self.get_f1_zeros()
            assert len(self.f1_zeros)>0, "No f1 zeros found! Change k-point grid"
        
        return

    #### Auxiliary functions

    def are_equal(self, v1=None, v2=None):
        """
        Just a function measuring the
        norm of the difference between two 1D numpy arrays.
        Returns True if the norm of the difference smaller
        or equal than a desired tolarance (default is 1e-5).                                                                                  
        
        :param v1: 3x1 numpy array (can be a python list)
        :param v2: 3x1 numpy array (can be a python list)
        """
        
        import numpy as np
        
        v1=np.array(v1)
        v2=np.array(v2)

        return np.linalg.norm(v1-v2)<=self.gen_tol
    ##
    
    def remove_dups(self, vect_list=None):
        """
        Just a function to remove duplicate
        vectors from a list of vectors 
        
        :param vect_list: nx3 numpy python list
        """

        out_list=[]
        for v in vect_list:
            
            if len(out_list)==0:
                out_list.append(v)
                
            elif len(out_list)>0:
                if any([ self.are_equal(v,vr) for vr in out_list]):continue
                out_list.append(v)
                
        return out_list
    ##

    def gen_kpts_grid_ns(self):
        """
        A function that determines the n1,n2,n3
        splits of the reciprocal unit vectors 
        for the k-point grid
        """

        import numpy as np
        
        b1=np.linalg.norm(self.rcell[0])
        b2=np.linalg.norm(self.rcell[1])
        b3=np.linalg.norm(self.rcell[2])
        
        step=(b1*b2*b3/self.nkpts)**(1./3)
        
        n1=int(round(b1/step))
        if np.mod(n1,2)==0: n1=n1+1
        n2=int(round(b2/step))
        if np.mod(n2,2)==0: n2=n2+1
        n3=int(round(b3/step))
        if np.mod(n3,2)==0: n3=n3+1
        
        if n1==0:n1=1
        if n2==0:n2=1
        if n3==0:n3=1
        
        return n1,n2,n3
    ##

    def get_kpts_grid(self):
        """
        A function to generate the k-point grid
        (as a list of k-points).
        """
        
        import itertools as itrtls
        import numpy as np
        
        n1,n2,n3=self.gen_kpts_grid_ns()

        kpts_grid=itrtls.product(np.linspace(0.,1.,n1),np.linspace(0.,1.,n2),np.linspace(0.,1.,n3))
        kpts_grid=[np.dot(self.rcell.T,gg) for gg in kpts_grid]

        return kpts_grid
    ##
        
    def get_R_stars(self):
        """
        A function to generate stars of
        R vectors in real space
        """
        
        import numpy as np
        import itertools as itrtls
        from operator import itemgetter
        
        # Generating R-vectors with crystal coordinates between -R_range and R_range (2*R_range+1 of them)
        # Hopefully we have more than 4 stars of R-vectors included.
        R_vect = [np.dot(self.cell.T,c) for c in itrtls.product(np.linspace(-self.R_range,self.R_range,2*self.R_range+1),repeat=3)]
        R_vect = [[r,np.linalg.norm(r)] for r in R_vect]
        R_vect = sorted(R_vect,key=itemgetter(1))
        
        ### separate R_vects into stars
        stars=[]
        
        for rr in R_vect:
            rr=list(rr[0])
            if len(stars)==0:
                star=[rr]
            elif any([rr in ss for ss in stars]):
                continue
            else:
                star=[rr]

            for rot in self.rotations:
                if not self.are_equal(rr,np.dot(rot,rr)):
                    star.append(list(np.dot(rot,rr)))

            star=self.remove_dups(star)
            stars.append(star)

        return stars
    ##

    ## now the functions to find zeros of and/or minimize
    def f1(self,x):
        """
        A function to generate the fully symmetric wave from R_star[1]
        """
        
        import numpy as np

        return np.sum([np.exp(-1.j*np.dot(S,x)) for S in self.R_stars[1]]).real
    ##
    
    def f2(self,x):
        """
        A function to generate the fully symmetric wave from R_star[2]
        """

        import numpy as np
        
        return np.sum([np.exp(-1.j*np.dot(S,x)) for S in self.R_stars[2]]).real
    ##

    def f3(self,x):
        """
        A function to generate the fully symmetric wave from R_star[3]
        """
        
        import numpy as np
        
        return np.sum([np.exp(-1.j*np.dot(S,x)) for S in self.R_stars[3]]).real
    ##
    
    def f4(self,x):
        """
        A function to generate the fully symmetric wave from R_star[4]
        """

        import numpy as np

        return np.sum([np.exp(-1.j*np.dot(S,x)) for S in self.R_stars[4]]).real
    ##
    
    ## Combining individual functions into vector functions to be able to
    ## use scipy.optimize.fsolve . It requires the argument and the function to be
    ## of the same length.
    
    # func1 for cases where there are zeros for the first three stars
    def func1(self,x):
        return [self.f1(x),self.f2(x),self.f3(x)]
    ##
    
    # func2 for cases where there are NO zeros for the first three stars
    def func2(self,x):
        return [self.f1(x),self.f2(x),0.]
    ##
    
    def get_f1_zeros(self):
        """
        A function to evaluate the k-points from the grid
        the are zeros of f1. These are then going to be used as 
        the initial guesses for the zeros of func1 and/or func2
        (using the scipy.optimize.fsolve).
        """
        
        import numpy as np
        
        print("Calculating f1_zeros, be patient.")
        print("There are %s k-points to go over." %(len(self.k_points)))
        
        ## find zeros of f1 on the k-grid to generate intial guesses
        ## for func1 or func2
        f1_zeros = [ii for ii in range(len(self.k_points)) if np.abs(self.f1(self.k_points[ii]))<self.zero]

        self.f1_zeros=f1_zeros
        
        return
    ##

    def __call__(self):
        """
        Sovling the actual equations to get the mean value point.
        """
        
        from scipy.optimize import fsolve
        import numpy as np
        
        if not hasattr(self,'f1_zeros'):
            self.get_f1_zeros()
        
        ## find zeros of func1 (simultaneous roots of f1 ,f2 and f3)
        ## by looping over the zeros of f1
        ## and using scipy.optimize.fsolve                                                                                                         
        zeros=[]

        for f1z in self.f1_zeros:
            kpt=self.k_points[f1z]
            root=fsolve(self.func1,kpt,xtol=self.zero)
            if np.linalg.norm(self.func1(root))<self.zero:
                zeros.append(root)

        # If there are solutions to func1=0
        # do some cleaning/arranging.
        # First we want to minimize f4 over those solutions.
        # Second, we do not care about zeros that are outside of
        # unit cell (in reciprocal cell).                                                                                                         
        if len(zeros)>0:
            
            # Minimize abs(f4) over the zeros of func1.
            # This is needed in case zeros contains multiple
            # symmetry not equvalent solutions
            data = [np.abs(self.f4(x)) for x in zeros]
            zeros = [zeros[ii] for ii in range(len(zeros)) if np.abs(data[ii]-min(data))<self.gen_tol]
            
            # bring those outside the unit cell back into the unit cell
            for ii in range(len(zeros)):
                zz=zeros[ii]
                zz_c=np.dot(np.linalg.inv(self.rcell.T),zz)
                if any(np.abs(zz_c)>=1.):
                    zz_c=zz_c-np.array([np.sign(zz_c[ii])*np.floor(np.abs(zz_c[ii])) for ii in range(3)])
                zeros[ii]=np.dot(self.rcell.T,zz_c)

            # remove possible duplicates
            zeros = self.remove_dups(zeros)
            assert len(zeros)>0, "No zeros found something is off!!!"

             
        elif len(zeros)==0:
            # If there are no solutions to func1=0
            # then find zeros of func2 (simultaneous roots of f1 and f2)
            # minimize f3 nad repeat the cleaning                                                                                         

            func2_zeros=[]
            
            for f1z in self.f1_zeros:
                kpt=self.k_points[f1z]
                root=fsolve(self.func2,kpt,xtol=self.zero)
                if np.linalg.norm(self.func2(root))<self.zero:
                    zeros.append(root)
            
            assert len(zeros)>0, "No zeros found something is off!!!"

            # Minimize abs(f3) over the zeros of func2
            data = [np.abs(self.f3(x)) for x in zeros]
            zeros = [zeros[ii] for ii in range(len(zeros)) if np.abs(data[ii]-min(data))<self.gen_tol]
            
            
            # bring those outside the unit cell back into the unit cell
            for ii in range(len(zeros)):
                zz=zeros[ii]
                zz_c=np.dot(np.linalg.inv(self.rcell.T),zz)
                if any(np.abs(zz_c)>=1.):
                    zz_c=zz_c-np.array([np.sign(zz_c[ii])*np.floor(np.abs(zz_c[ii])) for ii in range(3)])
                zeros[ii]=np.dot(self.rcell.T,zz_c)
                
            # remove possible duplicates
            zeros = self.remove_dups(zeros)

        if self.only_lowest_norm:
            norms=[np.round(np.linalg.norm(zz),6) for zz in zeros]
            return zeros[norms.index(min(norms))]

        else:
            return zeros
    ##
