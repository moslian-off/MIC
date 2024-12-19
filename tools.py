import matplotlib.pyplot as plt
import numpy as np
from main import GD_train,compute_entropy,iterative_approach_soft_clustering

def draw(P_C_i,i):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, len(P_C_i[i]), 1)
    plt.bar(x, P_C_i[i], width=0.8, color='skyblue', align='center')
    plt.xticks(x)
    plt.show()
    print(np.max(P_C_i[i]))

def draw_pij(i,j):
    pij_array = np.array(csr_matrix.toarray(pij[i,j]))
    x = np.arange(min, max+1, 1)
    y = np.arange(min, max+1, 1)

    xpos, ypos = np.meshgrid(x, y)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    zpos = np.zeros_like(xpos) 
    dx = dy = 0.4 
    dz = pij_array.flatten()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue')
    plt.show()
    
def GD_compute_entropy_with_different_T(N,Nc):
  entropy = []
  loss_array = []
  for t in range(75,90,1):
    temp_P_C_i = GD_train(4000,N,Nc,1.0/t,loss_array,lr=5e-4)
    entropy.append(compute_entropy(temp_P_C_i.detach().numpy()))
  return entropy

def iterative_approach_compute_entropy_with_different_T(N,Nc,similarity_matrix,begin,end):
  entropy = []
  for t in range(begin,end):
    P_C_i = iterative_approach_soft_clustering(similarity_matrix,1/t,Nc,1e-4)
    entropy.append(compute_entropy(P_C_i))
  # draw_entropy(entropy,begin,end)
  return entropy
