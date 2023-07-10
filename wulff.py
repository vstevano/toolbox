###############################
#
#  This file is written by Vladan Stevanovic
#
###############################

class Wulff():
    '''
    Class to construct Wulff shape and compute various other
    relevant quantities given the crystal structure
    and surface energies
    
    data: 
    Dictionary that contains Miller indices and
    associated surface energies in meV/A^2 
    (it will work if you provide sym inequivalent only)

    structure:
    Pylada stucture object containing the bulk structure 
    whose surface energies have been computed

    end_volume: default=1000. unit cells
    Volume that you would like the resulting grain to have
    '''
    
    def __init__(self,data=None,structure=None,end_volume=1000.):
        
        from pylada.crystal import read
        import numpy as np
        import spglib
        from toolbox.format_spglib import to_spglib

        #### Read input files

        self.data = data
        self.bulk = structure

        #### Setting up various variables

        self.dir_cell=np.transpose(self.bulk.cell)
        self.rec_cell=2*np.pi*np.transpose(np.linalg.inv(self.dir_cell))

        self.cryst_syms=spglib.get_symmetry(to_spglib(self.bulk),symprec=0.1)['rotations']
        self.cart_syms=[np.dot(self.dir_cell.T, np.dot(sym,np.linalg.inv(self.dir_cell.T)) ) for sym in self.cryst_syms]
        self.rcryst_syms=[np.dot(np.linalg.inv(self.rec_cell.T), np.dot(csym,self.rec_cell.T) ) for csym in self.cart_syms]
        
        self.end_volume=end_volume
        
        return

    ####
    # Function to not include duplicates in the star of v=np.dot(sym,(hkl))
    def in_star(self,vector,star):

        import numpy as np
        
        if len(star)==0:
            return False
        else:
            dd=[np.linalg.norm(vector-vv)<1e-5 for vv in star]

        if any(dd):
            return True
        else:
            return False

    ####
    # Function to assign surface energies to all hkl
    # not only the symmettry inequivalent ones
    
    def expand_surf_en(self):

        import numpy as np
        
        all_surf_en={}
        
        for key in self.data.keys():
            hkl=np.array(key.split(),dtype=int)
            
            star=[]
            for sym in self.rcryst_syms:
                vrot=np.dot(sym,hkl)
                if not self.in_star(vrot,star):
                    star.append(np.round(vrot,1))

            # Now, creating a dictionary with all hkl
            # as keys and cartesian vectors with their norms
            # equal to surface energy
            for ss in star:
                surf_en=self.data[key]
                pom=np.dot(self.rec_cell.T,ss)
                surf_en_vect=pom/np.linalg.norm(pom)*surf_en
                ss_hkl=str(ss)[1:-1]
                
                # Add to the dictionary
                all_surf_en[ss_hkl]=list(surf_en_vect)

        self.all_surf_en=all_surf_en
        
        return 

    ###
    # This was all preparation for the Wulff construction
    # Now, comes the Wulff construction

    def wulff_construct(self):
    
        from scipy.spatial import Voronoi, ConvexHull
        import numpy as np

        self.expand_surf_en()

        # First some cleaning so that we can follow what is going on (hopefully)
        
        gamma_n = list(self.all_surf_en.items())

        # Add the Gamma point as that is the one around which we will
        # draw the Voronoi region
        gamma_n.insert(0,['Gamma', np.array([0., 0.,  0.])])

        ############### EXTRACT THE VORONOI REGION AROUND THE GAMMA POINT

        gamma_n_hlp=np.array([x[1] for x in gamma_n])
        
        ## Creating a Voronoi object
        voronoi=Voronoi(gamma_n_hlp) 
        
        ## Finding the index of the origin (Gamma) point
        norms = [np.linalg.norm(x) for x in gamma_n_hlp]
        gamma_index = norms.index(0.)

        ## Getting all the vertices for all Voronoi shapes in Cart. coordinates
        verts = voronoi.vertices
        
        ## Getting the indices of the Voronoi region vertices around Gamma
        reg_indices = voronoi.regions[voronoi.point_region[gamma_index]]
        
        ## Extracting the Cart. coordinates of the vertices 
        ## belonging to the Voronoi region around Gamma (IBZ)
        reg = np.array([verts[ii] for ii in reg_indices])

        # Now compute the scale so that the output
        # matches desired end_volume
        if self.end_volume!=None:
            hull=ConvexHull(reg)
            volume=hull.volume
            scale=(self.bulk.volume*self.bulk.scale**3*self.end_volume/volume)**(1./3)
        else:
            scale=1.
            
        ## Get the faces of the IBZ (Cart. coordinates of the corners)
        ridges_indices=[]

        for key in voronoi.ridge_vertices:
            if all([x in reg_indices for x in key]):
                ridges_indices.append(key)         

        faces=[]
        
        for rdg in ridges_indices:
            faces.append([verts[ii]*scale for ii in rdg])
        
        return reg*scale,faces

    ####
    # Get the Miller indices, area, and the total surface
    # energy of a given face

    def get_face_info(self,face,miller_bounds=3):

        import numpy as np
        from itertools import product
        from copy import deepcopy
        from scipy.spatial import ConvexHull
        
        assert hasattr(self,'all_surf_en'), "Run the wulff.wulff_construct() first"
        
        # make it a np.array
        face=np.array(face)

        # center of mass
        cm=np.sum(face,axis=0)/len(face)

        # move the origin to the cm
        face=face-cm
        
        # Check are all cross products colinear
        norms=[np.cross(face[ii],face[jj])\
               for ii in range(len(face))\
               for jj in range(len(face)) if jj>ii]

        norms=[x for x in norms if np.linalg.norm(x)>1e-5]

        assert len(norms)>0, "There are very small faces, increase the end_volume"
        
        norms=np.array([nn/np.linalg.norm(nn) for nn in norms])

        check=np.array([np.cross(norms[ii],norms[jj])\
                        for ii in range(len(norms))\
                        for jj in range(len(norms)) if jj>ii]).flatten()

        assert all([x<1e-5 for x in check]), "Face is not flat"

        # Pick norms[0] (cross product between face[0] and face[1])
        # as the new z axis (verts are ordered counter clock, so...)
        # and the face[0] as the new x
        
        newz=norms[0]

        # Orient out just in case it is not already
        if np.dot(cm,newz)/np.linalg.norm(cm)<0.:
            newz=-newz
        
        newx=face[0]/np.linalg.norm(face[0])
        newy=np.cross(newz,newx)

        # Now get the Miller indices of the face
        for hkl_run,gamma in self.all_surf_en.items():

            projection=np.dot(gamma,newz)/np.linalg.norm(gamma)
            #print(hkl_run,projection)
            
            if 0.999<projection:
                hkl=deepcopy(hkl_run)
                break
            else:
                hkl=None

        assert hkl!=None, "Something is Off, cannot find hkl"

        
        # hkl done now the area
        
        # matrix to transform into the coordinate frame
        # where z is the normal
        mm=np.array([newx,newy,newz])

        new_face=np.array([np.dot(mm,ff) for ff in face])
        hull=ConvexHull(new_face[:,:2])
        area=hull.volume # area is volume in 2D

        # area done now the energy
        surf_en=np.linalg.norm(self.all_surf_en[hkl])

        # Making hkl and cartesian hkl np.arrays
        hkl=np.array(hkl.split(),dtype=float)
        hkl_cart=np.round(np.dot(self.rec_cell.T,hkl),4)

        return hkl, hkl_cart, area, area*surf_en

    ###
    # Get the total volume
    def get_tot_volume(self,reg):

        from scipy.spatial import ConvexHull

        hull=ConvexHull(reg)

        return hull.volume

    ###
    # Get the total area
    def get_tot_area(self,reg):

        from scipy.spatial import ConvexHull

        hull=ConvexHull(reg)

        return hull.area


    ###
    # Get the total surface energy
    def get_tot_surf_en(self,faces):

        import numpy as np

        return np.sum([self.get_face_info(face)[-1] for face in faces])


    ###
    # Plot only one face
    def plot_face(self, face):
        
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt
        from copy import deepcopy
        import numpy as np
        
        ## Initialize the figure 
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        ridges=deepcopy(face)
        ridges.append(ridges[0])
        ridges=np.array(ridges)
        ax.plot3D(ridges[:, 0], ridges[:, 1], ridges[:, 2], color='k')

        plt.show()
        
        return

    ####
    # Plot the Wullf shape
    def plot_wullf(self, reg=None, faces=None, figsize=[8,8],\
                   plot_corners=False, color='k', face_color=None,\
                   figname=None, view_angle=[45, 22.5], alpha=0.85,\
                   draw_xyz=True):
        
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from copy import deepcopy
        import numpy as np
        
        ############## DRAWING

        default_colors=['C%s' %(ii) for ii in range(10)]

        ## Initialize the figure with 1:1 aspect ratio
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')

        ## Draw the vertices of the IBZ
        if plot_corners:
            ax.scatter(reg[:, 0], reg[:, 1], reg[:, 2], marker='o', color=color)

        ## Draw the faces and edges of the IBZ
        for face in faces:

            ridges=deepcopy(face)
            ridges.append(ridges[0])
            ridges=np.array(ridges)
            ax.plot3D(ridges[:, 0], ridges[:, 1], ridges[:, 2], color=color, lw=3.)

            if face_color==None:
                ff = a3.art3d.Poly3DCollection([face],color=np.random.choice(default_colors), alpha=alpha)
            else:
                ff = a3.art3d.Poly3DCollection([face],color=face_color, alpha=alpha)
                
            ax.add_collection3d(ff)
            
            

        ##  Draw xyz 
        if draw_xyz:
            # axis limits
            minmax=[min(reg.flatten()),max(reg.flatten())]
            
            ax.plot3D([0.,minmax[1]], [0.,0.], [0.,0.], color='r')
            ax.plot3D([0.,0.], [0.,minmax[1]], [0.,0.], color='g')
            ax.plot3D([0.,0.], [0.,0.], [0.,minmax[1]], color='b')
        
            # Label each axis
            #ax.set_xlabel('x')
            #ax.set_ylabel('y')
            #ax.set_zlabel('z')
            
            # Set axis limits
            ax.set_xlim(minmax)
            ax.set_ylim(minmax) 
            ax.set_zlim(minmax) 
            
            
        ## Some figure plotting customization
        ax.set_axis_off()

        # Show rotating or savefig
        if figname==None:
            #ax.view_init(*view_angle)
            #plt.show()
            
            # rotate the axes and update
            for angle in range(0, 360):
                ax.view_init(view_angle[0]+angle, view_angle[1]+angle)
                plt.draw()
                plt.pause(.001)
        else:
            ax.view_init(*view_angle)
            plt.savefig(figname,dpi=600)
        
        return
