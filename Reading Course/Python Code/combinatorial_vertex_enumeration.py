import numpy as np
from itertools import combinations
import math
from fractions import Fraction
import nashpy as nash

def vertex_enumeration(A,B):
    m, n = A.shape
    tol = 1e-12
    
    # Normalize Payoff Matrices
    A = (A - A.min()) / (A.max() - A.min())
    B = (B - B.min()) / (B.max() - B.min())
     
    # Player 1 Vertices
    constraint_matrix = np.vstack((-1*np.eye(m),B.T))
    contstraint_values = np.concatenate([np.zeros(m),np.ones(n)])
    constraint_combos = list(combinations(range(n+m),m))
    num_combos = math.comb(n+m,m)
    
    vertex_strat_storage = np.zeros([num_combos,m])
    vertex_label_storage = np.zeros([num_combos,n+m])
    num_strats = 0
    for index in range(num_combos):
        constraint_combo = constraint_combos[index]
        submatrix = constraint_matrix[list(constraint_combo)]
        subvector = contstraint_values[list(constraint_combo)]
    
        try:
            player_1_vertex = np.linalg.solve(submatrix, subvector)
            player_1_vertex[np.abs(player_1_vertex)<tol] = 0
            eqn_values = constraint_matrix@player_1_vertex
                
            if np.all(player_1_vertex == 0):
                continue     
            if np.all(player_1_vertex >= 0) and np.all(eqn_values <= (contstraint_values+tol)):
                vertex_strat_storage[num_strats,:] = player_1_vertex
                vertex_label_storage[num_strats,list(constraint_combo)] = 1
                num_strats += 1
                
        except np.linalg.LinAlgError:
            continue 
    
    player_1_vertex_strategies = vertex_strat_storage[range(num_strats)]
    player_1_vertex_labels = vertex_label_storage[range(num_strats)]
    num_strats_1 = num_strats
    
    # Player 2 Vertices
    constraint_matrix = np.vstack((A, -1*np.eye(n)))
    contstraint_values = np.concatenate([np.ones(m), np.zeros(n)])
    constraint_combos = list(combinations(range(n+m),n))
    num_combos = math.comb(n+m,n)
    
    vertex_strat_storage = np.zeros([num_combos,n])
    vertex_label_storage = np.zeros([num_combos,n+m])
    num_strats = 0
    for index in range(num_combos):
        constraint_combo = constraint_combos[index]
        submatrix = constraint_matrix[list(constraint_combo)]
        subvector = contstraint_values[list(constraint_combo)]
    
        try:
            player_2_vertex = np.linalg.solve(submatrix, subvector)
            player_2_vertex[np.abs(player_2_vertex)<tol] = 0
            eqn_values = constraint_matrix@player_2_vertex
                
            if np.all(player_2_vertex == 0):
                continue     
            if np.all(player_2_vertex >= 0) and np.all(eqn_values <= (contstraint_values+tol)):
                vertex_strat_storage[num_strats,:] = player_2_vertex
                vertex_label_storage[num_strats,list(constraint_combo)] = 1
                num_strats += 1  

        except np.linalg.LinAlgError:
            continue 
    
    player_2_vertex_strategies = vertex_strat_storage[range(num_strats)]    
    player_2_vertex_labels = vertex_label_storage[range(num_strats)] 
    
    for player_1_strat in range(num_strats_1):
        for player_2_strat in range(num_strats):
            combined_labels = player_1_vertex_labels[player_1_strat,:] + player_2_vertex_labels[player_2_strat,:]
            if np.all(combined_labels==1):
                print("Nash Equilibrium!")
                p1 = player_1_vertex_strategies[player_1_strat,:]
                p2 = player_2_vertex_strategies[player_2_strat,:]
                p1 = p1/sum(p1) # Renormalize
                p2 = p2/sum(p2) # Renormalize
                p1 = [Fraction(x).limit_denominator() for x in p1]
                p2 = [Fraction(x).limit_denominator() for x in p2]
                print("p1: ",p1)
                print("p2: ",p2)
            
    return

## Example 1

A = np.array([
    [6,-2,3],   # Red = 1 vs Blue = (A,B,C,D)
    [-4,5,4],   # Red = 2 vs Blue = (A,B,C,D)    # Red = 3 vs Blue = (A,B,,DC)
], dtype=float)

B = -A # Zero Sum Game
                
vertex_enumeration(A,B)       
print("\n")

## Example 2

A = np.array([
    [1,25,26,2],   # Red = 1 vs Blue = (A,B,C,D)
    [35,1,36,3],   # Red = 2 vs Blue = (A,B,C,D)
    [45,46,1,4]    # Red = 3 vs Blue = (A,B,,DC)
], dtype=float)

B = -A # Zero Sum Game
                
vertex_enumeration(A,B)       
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")
   
## Example 3

A = np.array([
    [  6,   5,  -2,   0,  15],
    [  7,  -4,   3,   1,  17],
    [ -4,   2,   4,   5,   6],
    [  7,   2,   6,  -4,  17]
], dtype=float)

B = -A # Zero Sum Game
                
vertex_enumeration(A,B)         
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")
    
 ## Example 4
 
A = np.array([
    [2, 0, 1],
    [0, 1, 1]
], dtype=float)

B = np.array([
    [0, 2, -1],
    [3, 0,  0]
], dtype=float)

vertex_enumeration(A,B)        
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")
    
 ## Example 5
 
A = np.array([
    [-2, -7, -3],
    [-3, -2, 1],
    [1, -3, -2]
], dtype=float)

B = np.array([
    [-2, -3, 1],
    [1, -2,  -3],
    [-3,1,-2]
], dtype=float)

vertex_enumeration(A,B)        
            
game = nash.Game(A, B)
for sigma_1, sigma_2 in game.vertex_enumeration():
    sigma_1  =[Fraction(x).limit_denominator() for x in sigma_1]
    sigma_2  =[Fraction(x).limit_denominator() for x in sigma_2]
    print("Player 1:", sigma_1)
    print("Player 2:", sigma_2)
    print("\n")