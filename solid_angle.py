###############################
#
#  This file is written by Vladan Stevanovic
#
###############################

import numpy as np
#######################


class SolidAngle():
    '''
    Function to evaluate solid angle
    spanned by a polygon as viewed from a center point
    outside of it. This is done by splitting the
    pyramid into tetrahedra and then summing all 
    solid angles of individual tetrahedra
    '''

    def __init__(self,center=None, polygon=None):

        self.center=center

        # First if do we have a polygon at all?
        assert len(polygon)>=3, "Your polygon has only two corners, it is a line!!!"
        
        polygon=self.order_countercloc(polygon)
        self.polygon=polygon

        return
    ##########
    
    def get_solid_angle(self):
        '''
        center is a 1x3 array with coordinates of a point
        from which the polygon is looked at
        polygon is a nx3 array with coordinates 
        of its corners
        '''
    
        # Check whether the center and the polygon
        # are co-planar. In that case the result needs to be
        # 2*pi.

        from random import sample

        vectrs=self.polygon-self.center
        vectrs=[vectrs[ii] for ii in sample(range(len(vectrs)),3)]
        test_vol=np.abs(np.dot(vectrs[0],np.cross(vectrs[1],vectrs[2])))
        if test_vol<1e-5:
            return 2*np.pi
    
        # To form a tetrahedron one needs to successively pick three
        # points from the polygon. This is done by taking first three
        # then droping the first and adding the next and so on.
        # Critical thing is to have the polygon corners ordered
        # clockwise or counter clockwise.
        
        # Fourth, if the polygon is a triangle
        if len(self.polygon)==3:
            us=[plgn-self.center for plgn in self.polygon]
            return self.solid_angle_tetra(*us)

        # Fifth, if the polygon has more than 3 sides
        else:
            tetras = [[0]+list(range(ii,ii+2)) for ii in range(1,len(self.polygon)-1)]
        
            solid_angles=[]
        
            for tt in tetras:
                us=[self.polygon[ii]-self.center for ii in tt ]
                solid_angles.append(self.solid_angle_tetra(*us))
                
            return np.sum(solid_angles)
    ##########

    def solid_angle_tetra(self,u1,u2,u3):
        '''
        solid angle defined by three vectors
        u1, u2, and u3 which are not necessarily 
        normalized
        '''

        # Normalize them first
        u1=u1/np.linalg.norm(u1)
        u2=u2/np.linalg.norm(u2)
        u3=u3/np.linalg.norm(u3)
        
        # Absolute value is needed to remove the dependence
        # on the order of indices
        return np.abs(2*np.arctan( np.dot( np.cross(u1,u2), u3 )/\
                                   (1 + np.dot(u1,u2) \
                                    + np.dot(u1,u3) \
                                    + np.dot(u2,u3))))
    #########

    def order_countercloc(self,vertices):
        '''
        When splitting more complicated polyhedra into
        tetrahedra one needs to pay attention to the order
        of vertices so to not double count various regions.
        So, the first thing is to order indices according 
        to some rule. Seems that the ordering is provided
        by the ConvexHull function from Scipy. Nice!
        '''
        
        from operator import itemgetter
    
        # Find the center of the polygon
        mass_center=np.sum(vertices,axis=0)/len(vertices)
        
        # Make all position vectors in-plane nd normalized
        new_vertices=vertices-mass_center 
        new_vertices=np.array([x/np.linalg.norm(x) for x in new_vertices])
        
        # Calculate cross products between the in-plane vectors
        cross_prods=np.array([np.cross(new_vertices[0],vv) for vv in new_vertices])

        # check are all cross products colinear as they should
        col_test=[np.dot(cross_prods[ii],cross_prods[jj])/np.linalg.norm(cross_prods[ii])/np.linalg.norm(cross_prods[jj]) \
                  for ii in range(len(cross_prods)) for jj in range(ii,len(cross_prods))\
                  if np.linalg.norm(cross_prods[ii])>1e-5 and np.linalg.norm(cross_prods[jj])>1e-5 ]

        col_test=[np.round(np.abs(ct),3) for ct in col_test if np.round(np.abs(ct),3)>0.]
        assert all([ct==1. for ct in col_test]),\
            "Colinear test failed! Your vertices are not all in one plane!"

        # test finished
        ##############

        from scipy.spatial import ConvexHull

        # Change to a coor. frame wher one of the cross products
        # is new z-axis, one of the new_vertices x-axis
        # and their cross product new y-axis

        ex=new_vertices[0]
        
        ez=[cp for cp in cross_prods if np.linalg.norm(cp)>1e-3 and np.dot(cp,mass_center)>0.][0]
        ez=ez/np.linalg.norm(ez)

        ey=np.cross(ez,ex)

        # Transformation matrix to the new 
        m=np.array([ex,ey,ez])

        vertices_hlp=[np.dot(m,vv) for vv in vertices]
        
        xy_proj=np.array([[nv[0],nv[1]] for nv in vertices_hlp])
        hull=ConvexHull(xy_proj)
        
        # Use ConvexHull to order vertices
        index_counterclck=hull.vertices
        
        # Warn if the polygon is not convex
        if len(index_counterclck)<len(vertices): print("Your polygon is not convex!!!")
        
        # Finish
        return np.array([vertices[ii] for ii in index_counterclck])
    ##########

