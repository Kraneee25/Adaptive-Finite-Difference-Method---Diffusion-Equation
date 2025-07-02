# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:31:11 2024
@author: krane
"""
import  numpy as np
from dune.grid import cartesianDomain, Marker
from dune.alugrid import aluCubeGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import finiteVolume
import scipy.sparse as sparse
import time


class PoissonSolver:
    def __init__(self, a, f, Nx, Ny, I=None,
                 Lx=0, Rx=1, Ly=0, Ry=1):
        #-------input checkpoint-------#
        assert isinstance(a, (int,float))
        assert callable(f)
        assert isinstance(Nx, int)
        assert isinstance(Ny, int)
        assert Rx > Lx and Ry > Ly
        #---assigning class elements---#
        self.a = a
        self.I = I
        self.f = f
        self.Nx = Nx
        self.Ny = Ny

        self.Lx = Lx;   self.Rx = Rx;   self.Ly = Ly;   self.Ry = Ry
        #---initiating DUNE gridview---#
        self.domain = cartesianDomain([Lx,Ly], [Rx,Ry],
                                      [Nx, Ny])
        self.view = adaptiveLeafGridView( leafGridView( self.domain ) )
        self.indexSet = self.view.indexSet  # access index set from grid view

        self.space = finiteVolume(self.view)

        #---creating variables for storage later---#
        self.u_h = self.space.function(name='u_h')
        self.b = self.space.function(name='b')
        self.error = self.space.function(name='error')
        self.volume = self.space.function(name='volume')
        self.inner_point = self.space.function(name='inner_point')
        self.divergence = self.space.function(name='divergence')
        self.u = self.space.function(name='u')
        
    def details(self, u):
        print(f"Function: {u}")
        print(f"Domain:({self.Lx} x {self.Rx}) x ({self.Ly} x {self.Ry})")
        print(f"Degrees of freedom: {self.N}")
        print(f"Error: {self.globalerror}")
        print(f"Iterartion elapsed time: {round(self.it_time,2)}")
        print(f"Solution elapsed time: {round(self.sol_time,2)}")
        print(f"Total elapsed time: {round(self.it_time,2)+round(self.sol_time,2)}")
        print()

    def solver(self, markh=None, get_Details=False, central_difference = True):
        """
        The main function that solves the Poisson equation. This solver uses either the central difference
        method or the inner product method. 

        Parameters
        ----------
        markh : The condition to refine/coarsen the grid.
            This can be any function set by the user as long as the function has a grid element as an input and returns
            a Marker object.
        get_Details : Boolean
            Prints the necessary information of the problem. Mainly for comparing methods
        central_difference : Set True if this method is desired otherwise False for inner product method
        """


        if markh: #If a condition is set then refine/coarsen accordingly
            self.view.hierarchicalGrid.adapt(markh)
            self.view.hierarchicalGrid.loadBalance()
       
        "--Degrees of Freedom--"
        self.N=self.indexSet.size(0)
        
        "--Storage of values--"
        self.center = np.zeros((2,self.N))
        self.dx2Values = np.zeros((2,self.N))
        

        AValues = [[],[],[],[]]
        lineValues= [[],[],[],[]]
        posValues= [[],[],[],[]]
        
        start =time.time()
        
        
        "--Iteration of all of the elements of the grid--"
        for element in self.view.elements:
            k = self.indexSet.index( element ) # get element number
            
            xCenter,yCenter = element.geometry.center
            CurrentElementLvl = element.level
            volume= element.geometry.volume
            h = np.sqrt(volume) # or dy
            self.volume.as_numpy[k]=volume
            self.center[0,k]=xCenter; self.center[1,k]=yCenter # store coordinates to check error

            if get_Details:
                self.get_Details(element)
            
            self.b.as_numpy[k] = self.f(xCenter,yCenter)
            
            for intersection in self.view.intersections( element ):
                interIndex = intersection.indexInInside
                cs = (-1)**interIndex
                p = int(interIndex/2)
                
                if intersection.boundary:
                    AValues[interIndex].append(cs*1/h)
                    lineValues[interIndex].append(k)
                    posValues[interIndex].append(k)
                    
                    self.dx2Values[p,k]+=(h/2)
                
                    if self.I: #If given a boundary condition
                        if central_difference:
                            rel_dist = self.get_boundCoeff(element, p, CurrentElementLvl, h)
                            #print(xCenter-((1-p)*cs*dx),yCenter-(p*cs*dx) ,k)
                        else:
                            if interIndex==0 or interIndex==2:
                                AValues[interIndex].append(-1/h)
                                lineValues[interIndex].append(k)
                                posValues[interIndex].append(k+self.N)
                                
                                AValues[interIndex+1].append(1/h)
                                lineValues[interIndex+1].append(k+self.N)
                                posValues[interIndex+1].append(k)                               
                                      
                            rel_dist = h
                        self.b.as_numpy[k] += self.a*self.I(xCenter-((1-p)*cs*h),yCenter-(p*cs*h))/(h*rel_dist)
                        
                else:
                    neighbor = intersection.outside
                    j = self.indexSet.index( neighbor ) # get element number of neighbor
                    NeighborElementLvl = neighbor.level

                    if NeighborElementLvl == CurrentElementLvl: #same
                        AValues[interIndex].append(cs*-1/h)
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(j)
                        
                        AValues[interIndex].append(cs*1/h)
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(k)
                        
                        self.dx2Values[p,k]+=(h/2)  
                        
                    elif CurrentElementLvl < NeighborElementLvl: #smaller
                        AValues[interIndex].append(cs*-2/(3*h))
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(j)
                        
                        AValues[interIndex].append(cs*2/(3*h))
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(k)
                        
                        self.dx2Values[p,k]+=(3*h/16)
                        #self.dx2Values[p,k]+=(dx/4)
        
                    elif CurrentElementLvl > NeighborElementLvl: #bigger
                        q = self.get_NeighborIndex(k, neighbor, intersection.indexInOutside)
        
                        AValues[interIndex].append(cs*-2/(3*h))
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(j)
                        
                        AValues[interIndex].append(cs*1/(3*h))
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(k)
                        
                        AValues[interIndex].append(cs*1/(3*h))
                        lineValues[interIndex].append(k)
                        posValues[interIndex].append(q)  
                        
                        self.dx2Values[p,k]+=(3*h/4)
                        #self.dx2Values[p,k]+=(dx/2)
                        
        end = time.time()
        self.it_time=end-start
        start = time.time()        
        if central_difference:            
            matrix_shape=(self.N,self.N)
            self.backward_x = sparse.csr_matrix((AValues[0], (lineValues[0], posValues[0])), shape=matrix_shape)
            self.forward_x = sparse.csr_matrix((AValues[1], (lineValues[1], posValues[1])), shape=matrix_shape)
            self.backward_y = sparse.csr_matrix((AValues[2], (lineValues[2], posValues[2])), shape=matrix_shape)
            self.forward_y = sparse.csr_matrix((AValues[3], (lineValues[3], posValues[3])), shape=matrix_shape)

            self.sec_deriv_x = sparse.diags(np.reciprocal(self.dx2Values[0],where=(self.dx2Values[0]!=0))) @ (self.forward_x - self.backward_x) 
            self.sec_deriv_y = sparse.diags(np.reciprocal(self.dx2Values[1],where=(self.dx2Values[1]!=0))) @ (self.forward_y - self.backward_y)
            
            self.laplacian = self.sec_deriv_x + self.sec_deriv_y 
        
        else:
            self.backward_x = sparse.csr_matrix((AValues[0], (lineValues[0], posValues[0])), shape=(self.N,2*self.N))
            self.forward_x = sparse.csr_matrix((AValues[1], (lineValues[1], posValues[1])), shape=(2*self.N,self.N))
            self.backward_y = sparse.csr_matrix((AValues[2], (lineValues[2], posValues[2])), shape=(self.N,2*self.N))
            self.forward_y = sparse.csr_matrix((AValues[3], (lineValues[3], posValues[3])), shape=(2*self.N,self.N))

            self.laplacian = sparse.hstack((self.backward_x,self.backward_y))@(sparse.vstack((self.forward_x,self.forward_y)))
        
        self.A = -self.a*(self.laplacian) + sparse.identity(self.N)
        self.u_h.as_numpy[:] = sparse.linalg.spsolve(self.A, self.b.as_numpy)
        end = time.time()
        self.sol_time = end-start
        self.total_time=self.sol_time+self.it_time
        self.divergence.as_numpy[:] = abs((self.f(self.center[0],self.center[1])-self.u_h.as_numpy[:])/-self.a)
        self.minABSDiv = min(self.divergence.as_numpy)
        self.maxABSDiv = max(self.divergence.as_numpy)

    def get_trueError(self, true_solution):
        x = self.center[0]
        y = self.center[1]
        self.u.as_numpy[:] = true_solution(x,y)
        self.error.as_numpy[:] = ((self.u.as_numpy-self.u_h.as_numpy)**2)*(self.volume.as_numpy)
        self.globalerror=np.sqrt(sum(self.error.as_numpy[:]))
        return self.globalerror
    
    def get_numericalError(self):
        right_hand = self.f(self.center[0],self.center[1])
        left_hand = (self.A@self.u_h.as_numpy[:])+(self.b.as_numpy[:]-right_hand)
        self.residual = right_hand-left_hand
        print(self.residual)
        
    def get_Details(self,element):
        direction = ['W', 'E', 'S', 'N']
        k = self.indexSet.index( element ) # get element number
        print("Accessing element ", k)
        # obtaining element center coordinates
        center = element.geometry.center
        dx = np.sqrt(element.geometry.volume)
        print("center: ",center)
        print("dx: ", dx)
        for intersection in self.view.intersections( element ):
            neighbor = intersection.outside
    
            interIndex = intersection.indexInInside
            outerIndex = intersection.indexInInside
            orientation = direction[ interIndex ]
            if neighbor:
                j = self.indexSet.index( neighbor ) # get element number of neighbor
                print(f"Element {k} has neighbor {j} in direction {orientation} with center {neighbor.geometry.center} and edge in direction {direction[outerIndex]}")
            if intersection.boundary:
                print(f"Element {k} has boundary intersection in direction {orientation}")


    def grid_plot(self):
        self.view.plot()
        
    def true_plot(self, u):
        self.u.as_numpy[:]=u(self.center[0],self.center[1])
        self.u.plot()
        
    def f_plot(self):
        self.u.as_numpy[:]=abs(self.f(self.center[0],self.center[1]))
        self.u.plot()
        
    def a_plot(self):
        self.u.as_numpy[:]=self.A.diagonal()
        self.u.plot()
        
    def error_plot(self):
        self.error.plot()
        
    def div_plot(self):
        self.divergence.plot()

    def plot(self):
        self.u_h.plot()

    def get_NeighborIndex(self, elIdx, neighbor, indexInNeigh):

        for intersection in self.view.intersections( neighbor ):
            # search intersection with index provided
            if intersection.indexInInside == indexInNeigh:
                # access neighbor if exist
                outside = intersection.outside
                if outside:
                    idx = self.indexSet.index( outside )
                    # if index is same as element continue, we need the other intersection
                    if idx == elIdx:
                        continue
                    else:
                        return idx
        raise Exception("Not Found")
        
        
    def get_boundCoeff (self, boundary, p1, boundLvl, dx):
        for intersection in self.view.intersections( boundary ):
            interIndex = intersection.indexInInside
            p2 = int(interIndex/2)
            if not intersection.boundary:
                if p1 == p2:
                    neighbor = intersection.outside
                    NeighborElementLvl = neighbor.level
                    if boundLvl == NeighborElementLvl:
                        return dx
                    elif boundLvl < NeighborElementLvl:
                        return 0.5*dx+(3/8)*dx
            
        
        
    "---Class refinement Conditions premade by the Author---"
    def markh (self,element):
        k = self.indexSet.index(element)
        contri = (self.globalerror/self.N)*self.volume.as_numpy[k]
        if self.error.as_numpy[k] > contri:
            return Marker.refine
        else: 
            return Marker.coarsen
                
                
    def markhmax (self,element):
        k = self.indexSet.index(element)
        maxError = max(self.error.as_numpy)
        if np.allclose(self.error.as_numpy[k],maxError, rtol=1e-5):
            return Marker.refine
        else:
            return Marker.keep        
        
        
    def markhdiv (self,element):
        k=self.indexSet.index(element)
        div =  self.divergence.as_numpy[k]
        if np.allclose(div, self.maxABSDiv, rtol=0.5):
            return Marker.coarsen
        else:
            return Marker.refine
        
    def markhzero (self,element):
        k=self.indexSet.index(element)
        div =  self.divergence.as_numpy[k]
        if np.allclose(div, self.minABSDiv, rtol=50):
            return Marker.coarsen
        else:
            return Marker.refine
    
    def markhcenter (self,element):
        xCenter, yCenter = element.geometry.center
        if np.linalg.norm(np.array([xCenter,yCenter])-np.array([(self.Rx-self.Lx)/2, (self.Ry-self.Ly)/2]), ord=2) < 0.5:
            return Marker.refine
        else:
            return Marker.keep
        
    def markhcenter_rev (self,element):
        xCenter, yCenter = element.geometry.center
        if np.allclose(xCenter, (self.Rx-self.Lx)/2) and np.allclose(yCenter, (self.Ry-self.Ly)/2):
            return Marker.keep
        else:
            return Marker.refine
    
    def markhglobal (self,element):
        return Marker.refine


