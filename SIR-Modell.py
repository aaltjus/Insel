# -*- coding: utf-8 -*-
# Copyright 2016 Caren Tischendorf, Hella Rabus
#
# This work may be distributed and/or modified under the
# conditions of the LaTeX Project Public License, either version 1.3
# of this license or (at your option) any later version.
# The latest version of this license is in
#   http://www.latex-project.org/lppl.txt
# and version 1.3 or later is part of all distributions of LaTeX
# version 2005/12/01 or later.
#
# This work has the LPPL maintenance status `author-maintained'.
#
# This work consists of the files
#   bi_msv.py
#   bi_bsp_msv.py
#   bi_rk.py
#
# Erstellt von Caren Tischendorf, Hella Rabus
# fuer die VL "Angewandte Mathematik II"
#
#importiert die wichtigen Pakete
from numpy import array, exp, zeros, arange, mean
from scipy.linalg import solve
from scipy.optimize import fsolve
from matplotlib import use
use('TkAgg')
from matplotlib.pyplot import (axis,
            close,
            legend,
            plot,
            rcParams,
            show,
            title,
            xlabel,
            ylabel,
            yscale,
            xscale)
from pylab import savefig
import math

# define a general solver for Runge Kutta methods
class RKsolve(object):
    # empty constructor
    def __init__(self):
        pass

    # numerical integration phase of Runge Kutta
    # t0 initial time point
    # T  final time point
    # x0 initial solution point
    # h  stepsize
    # N number of time intervals
    def integrate(self, t0, T , x0, h):
        self.h = h
        dim = len(x0)
        self.dim = dim
        s = self.snum # number of Runge Kutta stages
        print "dim=", dim
        #Anzahl der Schritte
        N = int((T-t0)/h)
        self.N = N
        print "N=", N
        # initialize time and space variables
        tout=zeros(N+1)
        xout=zeros((N+1,dim))
        tout[0] = t0
        xout[0,:] = x0
        xold = x0
        told = t0
 
        # approximate x(t_n+1)
        yinitial = zeros(s*dim)
        for n in xrange(1,N+1):
            for i in xrange(s):
            # compute initial solution for Newton iteration
                yinitial[i*dim:(i+1)*dim] = self.fdgl(xout[n,:],tout[n])
            y = fsolve(self.fRK,yinitial,args=(xold,told),xtol=1e-8, maxfev=50)
            xn = xold
            for j in xrange(s):
                Yj = y[j*dim:(j+1)*dim]
                xn = xn + h*self.RKb[j]*Yj  # Approximation von x(t_n)
            tout[n] = told + h
            xout[n,:] = xn
            xold = xn
            told = told + h
        self.xout = xout
        self.tout = tout

#######################################################

    # t current time point
    # xold previous solution point x_{n-1}
    # describes the RK equation for ODE systems fRK(y)=0 ODE system=gewöhnliches Diff-gleichungs-system
    # with y containing all stage solutions Y_i
    def fRK(self,y,xold,told):
        h = self.h
        RKA = self.RKA
        RKb = self.RKb
        RKc = self.RKc
        s = self.snum
        dim = len(xold)
        Y = zeros((s,dim))
        for i in xrange(s):
            Y[i,:] = y[i*dim:(i+1)*dim]  # Stufenapproximation Y_i
        fRK = zeros(s*dim)
        hRKA = h*RKA
        for i in xrange(s):
            ti = told + RKc[i] * h  # Zwischenzeitpunkte t_i
            xi = xold
            for j in xrange(s):
                xi = xi + hRKA[i,j] * Y[j,:]  # Approximation von x(t_i)
            fRK[i*dim:(i+1)*dim] = Y[i,:] - self.fdgl(xi,ti)
        return fRK

########################################################

    # describes different Runge Kutta methods of different stages/orders, implicit/explicit
    # method  string encoding the method to be used by the solver
    def coeffRK(self, method):
        # number of Runge Kutta stages
        #s = self.snum
        if method == 'implEuler':
            self.snum = 1
            A = [[1.]]
            b = [1.]
            c = [1.]
        if method == 'RK4':
            # this method is of convergence order = 4
            # number of stages 
            self.snum = 4
            A = [ [0., 0., 0., 0.],
            [0.5, 0., 0., 0.],
            [0. ,0.5 ,0. ,0.],
            [0. ,0. ,1. ,0] ]
            b = [1./6, 1./3, 1./3, 1./6]
            c = [0., 0.5, 0.5, 1.]
        if method == 'RungeKutta':
            #unser Löser mit Butcher-Tableau der Größe 2 (siehe unten), das gamma aus der Aufgabe haben wir hier delta genannt
            self.snum = 2
            delta = (3-((3)**(1./2)))/6
            A = [[delta, 0],
                 [1-2*delta, delta]]
            b = [1./2,1./2]
            c = [delta, 1-delta]
        #setzt Eigenschaften von self, fügt die entsprechenden arrays hinzu
        self.RKA = array(A)
        self.RKb = array(b)
        self.RKc = array(c)
       
        return

########################################################

    # describes the right hand side of the ODE to be solved
    def fdgl(self, x ,t):
        example = self.example
        if  example == 'bspBI': 
            y = -x
        if example == 'SIR':
            y = zeros(3)
            #y[0] = S'
            y[0] = -self.beta*x[0]*x[1]
            #y[1] = I'
            y[1] = self.beta*x[0]*x[1]-self.gamma*x[1]
            #y[2] = R'
            y[2] = self.gamma*x[1]
        return y
    
########################################################

    # implements the analytical solution of the ODE described above
    def exactsol(self):
        t = self.tout
        N = len(t)
        example = self.example
        if  example == 'bspBI':
            x = zeros((N,1))
            for n in xrange(N):
                x[n,0] = exp(-t[n])             
        self.xexact = x

########################################################

    # plot the approximate solution to a file with a discriptive naming
    def plot(self,i):
        # plot i-th component
        t = self.tout
        x = self.xout[:,i]
        #ordnet den jeweiligen Stellen im Vektor die richtige Bezeichnung zu, die vom Plot aufgerufen werden kann
        label = ['Anzahl der Gesunden', 'Anzahl der Kranken', 'Anzahl der Resistenten']
        p0 = plot(t, x, linewidth = 2, label = label[i])
        try:
            self.exactsol()
            p1=plot(t,self.xexact, 'r',label='exakte Loesung')
        except:
            print("no exact solution given")
        title('Beispiel '+self.example+' mit '+self.method+', Stufenanzahl:'+str(self.snum))
        rcParams.update({'font.size': 12})
        legend(loc='center right', shadow=True)
        xlabel('t')
        ylabel('x(t)')
        savefig('sol_'+self.example+'_'+self.method+'_s'+str(self.snum)+'_i'+str(i)+'.png')
        #print (t[0], t[1e2], t[2*1e2], t[3*1e2], t[4*1e2]), (x[0], x[1e2], x[2*1e2], x[3*1e2], x[4*1e2])
        #show()
        #close()
        
########################################################

#hier werden beta und gamma automatisiert berechnet
def parameterbestimmung():
    expected = [3.0, 10., 30., 65., 93.]
    bestebetaundgamma = []
    besteabweichung = 20
    for beta in arange(0.003, 0.006, 0.001):
        for gamma in arange(0.4, 0.9, 0.05):
            print 'trying beta: {}, gamma: {}'.format(beta, gamma)
            bi_solve = RKsolve()
            bi_solve.example = 'SIR'  
            bi_solve.method = 'RungeKutta'
            bi_solve.beta = beta
            bi_solve.gamma = gamma
            if bi_solve.example == 'SIR':
                T = 20.
                t0 = 1
                x0 = array([397.0, 3.0, 0.0])
                h = 1e-2
            bi_solve.coeffRK(bi_solve.method)
            bi_solve.integrate(t0,T,x0,h)
            #setzt x so, dass in der Liste nur noch die Kranken enthalten sind
            x = bi_solve.xout[:,1]
            t = bi_solve.tout
            #setzt kranke als Liste von gemessener Anzahl von Kranken an den Stellen 1,2,3,4,5
            kranke = [x[0], x[1e2], x[2*1e2], x[3*1e2], x[4*1e2]]
            #bestimmt die Abweichung von gemessenen Werten zu den erhofften (expected) nach Woche 1,2,3,4 und 5
            differenzen=[]
            for i in range (0, 5):
                differenzen.append(expected[i] - kranke[i])
            #bestimmt den Betrag (abs) der mittleren Abweichung
            mittlereabweichung = abs(mean(differenzen))
            #ersetzt die beste Abweichung mit dem aktuell besten Wert und speichert die zugehörigen beta und gamma
            if mittlereabweichung<besteabweichung:
                besteabweichung = mittlereabweichung
                bestebetaundgamma = [beta, gamma]
    print ('Die geringste mittlere Abweichung beträgt '+str(besteabweichung)+'. Die Parameter dazu lauten: beta = '+str(bestebetaundgamma[0])+' gamma = '+str(bestebetaundgamma[1]))
            
                
########################################################
         
# run the main program
if __name__ == "__main__":
    import sys
    parameterbestimmung()
    sys.exit(0)
    # initialize a Runge Kutta solver
    bi_solve = RKsolve()
    bi_solve.example = 'SIR'  
    bi_solve.method = 'RungeKutta'
    bi_solve.beta = 0.004
    bi_solve.gamma = 0.4
    #bi_solve.snum = 4 # number of Runge Kutta stages to be used, if not set by method itself
    
    if  bi_solve.example == 'bspBI':
        T = 2.0
        t0 = 0.0
        x0 = array([1.0])
        h = 1e-2
    if  bi_solve.example == 'bspBstart':
        T = t0+2*h
        t0 = 0.0
        x0 = array([1.0])
        h = 1e-2   
    if bi_solve.example == 'SIR':
        T = 20.
        t0 = 1
        x0 = array([397.0, 3.0, 0.0])
        h = 1e-2
    
    bi_solve.coeffRK(bi_solve.method)
    # actual solve phase
    dim = len(x0)
    
    # Konvergenzordnungstest
    #hlist = [h, h/2.0, h/4., h/8., h/16., h/32, h/64, h/128, h/256, h/512, h/1024, h/2048, h/4096]
    #hlist = [2**(-i) for i in xrange(12)]
    #bi_solve.konvergenzordnungsplott(t0,T,x0,hlist)
    bi_solve.integrate(t0,T,x0,h)
    
    #nur für unsere DGL, also SIR, auskommentiert, da wir keine exakte Lösung haben
    #bi_solve.exactsol()
    #t = bi_solve.tout
    #for k in xrange(dim):  # calculate error for k-th component
         #xexactk = bi_solve.xexact[:,k]
         #errork = (abs(bi_solve.xout[:,k]-xexactk))
         #for j in xrange(len(t)):
            #errork[j] = errork[j]/max(abs(xexactk[j]),1e-40)

    #print bi_solve.xout
    # plotting
    if bi_solve.example == 'bspBI':
        bi_solve.plot(0)
        show()
        close()
    if bi_solve.example == 'SIR':
        bi_solve.plot(0)
        bi_solve.plot(1)
        bi_solve.plot(2)
        show()
        close()
   


