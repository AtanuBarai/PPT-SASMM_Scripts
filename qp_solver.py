#!/usr/bin/env python3
#
#Author: Nandakishore Santhi <nsanthi@lanl.gov>
#Date: 10 August 2019
#Purpose: Generate multivariate polynomial fits for data points using globally convergent, optimal, convex optimization
#Copyright: Los Alamos National Laboratory. Part of the PPT open source code release.
#
"""
Quadratic Programming solver for the SNAP input data
"""
import sys
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import sympy as sp
import cvxpy as cp
import math
#from numba import jit

#List all global variables here:
x_values = np.array([1024, 1024])
x_symbols = sp.symbols('Nx, Ny, Nz, Ichunk, Nmom, Nang, Ng')
x_symbols = sp.symbols('x1, x2')

maxDeg = len(x_symbols)
maxTerms = 12 #To consider for pruning after 1st convex optimization stage
kernelBB = False #If true, does fit for the most prominent basic-block
singleBB = False #If true, does fit for below given basic-block# only, other wise for all BBs
#BB_col = 342
#BB_col = 321

#Convex optimizer parameters, verbosity etc:
maxIters = 1000 #Base iterations, gets multiplied by maxDeg
eps = 1e-15
alphaMin = 1e-5
verbose = False

def getData(filename):
    dataFile = pd.read_csv(filename, delimiter='\s+', header=None)
    data = dataFile.values
    if verbose: print(data.shape)
    return data

#@jit(nopython=True)
def prepAllMonomialCols(data):
    M = sorted(sp.polys.monomials.itermonomials(x_symbols, maxDeg), key=sp.polys.orderings.monomial_key('grlex', x_symbols)) #Monomial list
    Zarray = [] # The Z values
    vDict = {}
    for i in range(data.shape[0]):
        dataRow = data[i]
        for k in range(len(x_symbols)): vDict[x_symbols[k]] = dataRow[k]
        Zrow = []
        for m in M:
            P = sp.Poly(m, x_symbols)
            v = P.eval(vDict)
            Zrow.append(v)
        Zarray.append(Zrow)
        print(".", end="")
    print("")
    Z = np.asarray(Zarray)
    np.save('Zvals_' + str(maxDeg), Z)
    return Z

#@jit(nopython=True)
def convexOpt(data, Z, BB_col, normalize=False, index=[], eps=1e-20, alphaMin=1e-3, maxIters=3000, tag=""):
    """
    This is the step where the quadratic programing problem is solved
    """
    print("Perform convex optimization " + str(tag) + "...")

    normalizationWeights = np.mean(Z, axis=0)
    if verbose: print("Column weights:", normalizationWeights)

    if normalize:
        Z_n = Z/normalizationWeights
    else:
        Z_n = Z

    y = data[:, BB_col] # Target values for prediction, BB in column

    constraints = []
    if len(index) > 0: #Sorted index given
        if verbose: print("Sorted index list is given!")

        nullDict = {}
        for i in range(Z.shape[1]): nullDict[i] = True
        for i in index: nullDict[i] = False

        nullList = []
        for i in nullDict:
            if nullDict[i]: nullList.append(i)

        Z_n = np.delete(Z_n, nullList, axis=1)
        normalizationWeights = np.delete(normalizationWeights, nullList)

    alpha = cp.Variable(Z_n.shape[1])
    if verbose: print("Z shape {}, alpha shape {}, output y shape {}".format(Z_n.shape, alpha.shape, y.shape))

    constraints = [alpha >= alphaMin]

    try:
        prob = cp.Problem(cp.Minimize(cp.norm(Z_n@alpha-y)), constraints)
        prob.solve(solver=cp.SCS, verbose=verbose, eps=eps, max_iters=maxIters)
    except:
        print("Convex solver failed!")

    if prob.value:
        print("Found a Global Minimum {}".format(prob.value))
        return prob.value, alpha.value, normalizationWeights
    else:
        print("Failed to find a Global Minimum (most likely due to lower than required precision settings)")
        return None, None, None

def nestedOpt(data, Z, BB_col, normalize=False, eps=1e-20, alphaMin=1e-3, maxIters=3000, tag=""):
    probVal0, alphaVal0, normalizationWeights0 = convexOpt(data, Z, BB_col, normalize=True, eps=1e-8, alphaMin=alphaMin, maxIters=3000, tag="(BB# " + str(BB_col) + ", #1, Full)")

    if probVal0:
        if verbose:
            print("\n(BB# = " + str(BB_col), end="):> ")
            for i in range(len(alphaVal0)):
                if alphaVal0[i] != 0.0:
                    if alphaVal0[i] > 0.0: print(" + ", end="")
                    else: print(" - ", end="")
                    print(str(math.fabs(alphaVal0[i])), end="")
                    if i != 0: print("*(" + str(M[i]) + ")", end="")
            print("\n")

        #Solve again after pruning the polynomial to just a few significant terms
        alphaAbs0 = np.absolute(alphaVal0)
        alphaIndSorted0 = np.sort(np.argsort(alphaAbs0)[-maxTerms:]) #Take only last maxTerms
        if verbose: print("Sorted alpha indices:", alphaIndSorted0)
        probVal, alphaVal, normalizationWeights = convexOpt(data, Z, BB_col, normalize=False, index=alphaIndSorted0, eps=eps, alphaMin=alphaMin, maxIters=maxIters, tag="(BB# " + str(BB_col) + ", #2, Pruned)")

        if probVal:
            maxVal = abs(max(alphaVal, key=abs))
            print("\n(BB# = " + str(BB_col), end="):> ")
            for i in range(len(alphaVal)):
                if (abs(alphaVal[i]) > maxVal/6) and (alphaVal[i] != 0.0):
                    if alphaVal[i] > 0.0: print(" + ", end="")
                    else: print(" - ", end="")
                    print(str(math.fabs(alphaVal[i])), end="")
                    if i != 0: print("*(" + str(M[alphaIndSorted0[i]]) + ")", end="")
            print("\n")
        sys.stdout.flush()

if __name__ == "__main__":
    data = getData(sys.argv[1]) #Load data to fit to multivariate polynomials

    if len(sys.argv) > 2: maxDeg = int(sys.argv[2])
    M = sorted(sp.polys.monomials.itermonomials(x_symbols, maxDeg), key=sp.polys.orderings.monomial_key('grlex', x_symbols)) #Monomial list

    ZvalFileName = "Zvals_" + str(maxDeg) + ".npy"
    ZvalFile = Path(ZvalFileName)
    print("Checking if file " + ZvalFileName + " exists?", end=" ")
    if ZvalFile.is_file():
        print("Yes!")
        Z = np.load(ZvalFile, allow_pickle=True)
    else:
        print("No!")
        print("Creating file " + ZvalFileName + "!")
        Z = prepAllMonomialCols(data)

    BB_col_min, BB_col_max = len(x_symbols), data.shape[1] #default range
    if kernelBB:
        result = np.where(data == np.amax(data))
        listOfCordinates = list(zip(result[0], result[1]))
        BB_col = listOfCordinates[0][1]
        BB_col_min, BB_col_max = BB_col, BB_col+1
    elif singleBB:
        BB_col_min, BB_col_max = BB_col, BB_col+1

    print("BB# selected in range [", BB_col_min, ":", BB_col_max, ")\n")
    maxIters *= maxDeg
    for BB_col in range(BB_col_min, BB_col_max):
        try:
            nestedOpt(data, Z, BB_col, normalize=False, eps=eps, alphaMin=alphaMin, maxIters=maxIters, tag="")
        except:
            print("Encountered error at BB# " + BB_col)
