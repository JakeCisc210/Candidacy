import numpy as np
from itertools import combinations
import math
from fractions import Fraction
import nashpy as nash

def manual_support_enumeration(A,B):
    m, n = A.shape
    max_support = min(m,n)
    for k in range(1,max_support+1):
        player_1_supports = list(combinations(range(m),k))
        player_2_supports = list(combinations(range(n),k))
        
        num_S1s = math.comb(m,k)
        num_S2s = math.comb(n,k)
        
        for index in range(num_S1s*num_S2s):
            
            S1_index = index % num_S1s
            S2_index = index // num_S1s
            
            S1 = player_1_supports[S1_index]
            S2 = player_2_supports[S2_index]                                 
                                     
            reduced_A = A[np.ix_(S1, S2)]
            reduced_B = B[np.ix_(S1, S2)]
            
            # Solve Indifference Equations
            
            # [ reduced_B^T -1 ] [p^1]  = [0]
            # [ 1^T          0 ] [μ]     [1]
            
            ones = np.ones(k)
            matrix_1 = np.block([[reduced_B.T,-ones[:, None]],
                [ones[None, :], np.zeros((1, 1))]])
    
            # [ reduced_A   -1 ] [p^2]  = [0]
            # [ 1^T          0 ] [λ]     [1]
            
            matrix_2 = np.block([[reduced_A,-ones[:, None]],
                [ones[None, :], np.zeros((1, 1))]])  
            
            vector = np.concatenate([np.zeros(k), np.array([1.0])])
            
            try:
                solution_1 = np.linalg.solve(matrix_1, vector)
                solution_2 = np.linalg.solve(matrix_2, vector)
            except np.linalg.LinAlgError:
                continue 
            
            p1 = np.zeros(m)
            p2 = np.zeros(n)
            
            for index in range(k):
                p1[S1[index]] = solution_1[index]   
                p2[S2[index]] = solution_2[index]    
                            
            mu = solution_1[k]
            lam = solution_2[k]
            
            # Check for Nash Equilibrium
            nash_equilibrium = np.all(p1>=-1e-10) and np.all(p2>=-1e-10)
                
            if nash_equilibrium:
                expected_values_1 = A@p2
                for strategy in [index for index in range(m) if index not in set(S1)]:
                    if expected_values_1[strategy] > lam + 1e-10:
                        nash_equilibrium = 0                  
                        break
                    
            if nash_equilibrium:
                expected_values_2 = p1@B
                for strategy in [index for index in range(n) if index not in set(S2)]:
                    if expected_values_2[strategy] > mu + 1e-10:
                        nash_equilibrium = 0
                        break

            if nash_equilibrium:
                print("Nash Equilibrium!")
                p1 = [Fraction(x).limit_denominator() for x in p1]
                p2 = [Fraction(x).limit_denominator() for x in p2]
                print("p1: ",p1)
                print("p2: ",p2)
                return

## Example 1

A = np.array([
    [1,25,26,2],   # Red = 1 vs Blue = (A,B,C,D)
    [35,1,36,3],   # Red = 2 vs Blue = (A,B,C,D)
    [45,46,1,4]    # Red = 3 vs Blue = (A,B,,DC)
], dtype=float)

B = -A # Zero Sum Game
                
manual_support_enumeration(A,B)       
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")
   
## Example 2

A = np.array([
    [  6,   5,  -2,   0,  15],
    [  7,  -4,   3,   1,  17],
    [ -4,   2,   4,   5,   6],
    [  7,   2,   6,  -4,  17]
], dtype=float)

B = -A # Zero Sum Game
                
manual_support_enumeration(A,B)       
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")
    
 ## Example 3
 
A = np.array([
    [2, 0, 1],
    [0, 1, 1]
], dtype=float)

B = np.array([
    [0, 2, -1],
    [3, 0,  0]
], dtype=float)

manual_support_enumeration(A,B)       
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")