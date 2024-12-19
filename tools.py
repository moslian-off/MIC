import matplotlib.pyplot as plt
import numpy as np

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