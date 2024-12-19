import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm
import torch


def load_data():
  df = pd.read_excel("data/TicketData.xlsx")
  category = df["Sector"]
  change_ratio_list = df["ChangeRatioList"]
  change_ratio_array = []

  for i in range(N):
    change_ratio_array.append(np.array(eval(change_ratio_list[i])).astype(int)) 
    change_ratio_array = np.array(change_ratio_array)

  return change_ratio_array,category

def to_pi(change_ratio_array):
    N,Days = change_ratio_array.shape
    result = []
    for ticketer in tqdm(range(N)):
      count_array = np.zeros(max-min+1)
      for value in change_ratio_array[ticketer]:
        if min <= value <= max:
          count_array[value-min] += 1 
        
      p_i = count_array / Days
      result.append(p_i)    
    result = np.array(result)
    result[np.where(result <= 0)] = 1e-10
    return np.array(result)

def to_pij(change_ratio_array):
    N = len(change_ratio_array)
    min_val = np.min(change_ratio_array)
    max_val = np.max(change_ratio_array)
    
    result = np.empty((N, N), dtype=object)
    
    elements = max_val - min_val + 1
    counter = np.zeros((elements, elements))
    
    for i in tqdm(range(N)):
        offset_i = change_ratio_array[i] - min_val
        for j in range(i, N):
            offset_j = change_ratio_array[j] - min_val
            for idx_i, idx_j in zip(offset_i, offset_j):
                counter[idx_i][idx_j] += 1

            if np.sum(counter) > 0:
              counter /= np.sum(counter) 
        
            counter_sparse = csr_matrix(counter)
            result[i, j] = counter_sparse
        
            if i != j:
              result[j, i] = counter_sparse.T
    
    return result

def MI(pi, pj, pij):
    N = len(pij)
    result = np.zeros((N, N))
    epsilon = 1e-10 

    for i in tqdm(range(N)):
        for j in range(i, N):  
            pi_array = np.array(pi[i])
            pj_array = np.array(pj[j])
            pij_array = csr_matrix.toarray(pij[i, j])
            
            pi_array = np.where(pi_array == 0, epsilon, pi_array)
            pj_array = np.where(pj_array == 0, epsilon, pj_array)
            pij_array = np.where(pij_array == 0, epsilon, pij_array)

            temp_info = np.sum(pij_array * np.log2(pij_array / np.outer(pi_array[:, None] , pj_array[None, :])))

            result[i, j] = temp_info
            if i != j:
                result[j, i] = temp_info
    
    return result

def normalize(mutual_info):
    D = np.diagonal(mutual_info)
    D_prime = 1 / np.sqrt(D)
    D_prime_matrix = np.diag(D_prime)
    normalized_mutual_info = D_prime_matrix @ mutual_info @ D_prime_matrix
    return normalized_mutual_info

def main():
  change_ratio_array,category = load_data()
  pi = to_pi(change_ratio_array)
  pij = to_pij(change_ratio_array)
  pearson_matrix = np.corrcoef(change_ratio_array)
  np.savetxt("Pearson_matrix_tickets.txt",pearson_matrix,delimiter=",")
  mutual_info = MI(pi, pi, pij)
  mutual_info = normalize(mutual_info)


  
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(pearson_matrix)
plt.colorbar()
plt.savefig('imgs/pearson_matrix_ticket.png')


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

draw_pij(0,2)

np.savetxt("temp/ticket_sij.csv",mutual_info,delimiter=",")

plt.figure(figsize=(8, 6))
plt.imshow(mutual_info)
plt.colorbar()
plt.show()

np.savetxt("temp/normalized_ticket_sij.csv",mutual_info,delimiter=",")

# %% [markdown]
# #### Gradient Descent Method

# %%

# mutual_info = torch.tensor(normalize(pearson_matrix))
mutual_info = torch.tensor(np.loadtxt("temp/normalized_ticket_sij.csv", delimiter=","))
# mutual_info = torch.tensor(np.loadtxt("temp/ticket_sij.csv", delimiter=","))

# %%
def compute_P_C(P):
    return torch.sum(P,dim=0)/N

def compute_s_C(P, similarity_matrix, C):
    P_C = compute_P_C(P)
    P_column = P[:, C]
    norm_factor = P_C[C] ** 2
    s_C = torch.sum(similarity_matrix * torch.ger(P_column, P_column)) / norm_factor/(N**2)
    return s_C

# %%
def target_fc(P_C_i_temp, T):  
    epsilon = 1e-10
    # P_C_i_normalized = P_C_i / P_C_i.sum(dim=1, keepdims=True).clamp(min=1e-10)
    # P_C_i = P_C_i_normalized.clamp(min=epsilon)
    P_C_i = P_C_i_temp/torch.sum(P_C_i_temp,dim = 1,keepdim = True)  
    P_C = compute_P_C(P_C_i)
    
    N, Nc = P_C_i.shape
    s_C = torch.zeros(Nc)
    for i in range(Nc):
      s_C[i] = compute_s_C(P_C_i, mutual_info, i)
    s_all_clusters = torch.dot(s_C, P_C)
    info = torch.sum(P_C_i * torch.log2(P_C_i / P_C)) / N
    return -s_all_clusters + T * info

def train(epoch,N,Nc,T,loss_array = None,lr = 1e-3):
    P_C_i = torch.rand(N,Nc,requires_grad=True)
    optimizer = torch.optim.Adam([P_C_i],lr)

    for i in range(epoch):
        optimizer.zero_grad()
        loss = target_fc(P_C_i,T)
        loss.backward()
        optimizer.step()
        print(f"Epoch:{i}, loss:{loss.item()}")
        if loss_array is not None:
          loss_array.append(loss.item())
    if loss_array is not None:
      loss_array = np.array(loss_array)
    return torch.softmax(P_C_i,dim=1)

# %%
T = 1/60
Nc = 11
N = 496
     
import copy

loss_array = []    
P_C_i = train(5000,N,Nc,T,loss_array,lr=1e-2)

np.savetxt("temp/ticket_opt_P_C_i.csv",P_C_i.detach().numpy(),delimiter=",")

# %%
import matplotlib.pyplot as plt

x = np.arange(0,5000,1)
plt.plot(x,loss_array)
plt.ylabel("target")
plt.savefig("imgs/ticket_loss.png")
P_C = compute_P_C(P_C_i)
P_C

# %%
P_C_i = np.loadtxt("temp/ticket_opt_P_C_i.csv",delimiter=",")

def draw(P_C_i,i):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, len(P_C_i[i]), 1)
    plt.bar(x, P_C_i[i], width=0.8, color='skyblue', align='center')
    plt.xticks(x)
    plt.show()
    print(np.max(P_C_i[i]))
    
draw(P_C_i,0)

# %%
# entropy = []

# def compute_entropy(P_C_i):
#     return -np.sum(P_C_i * np.log2(P_C_i)) if np.all(P_C_i > 0) else 0
 
# for t in range(75,90,1):
#   temp_PCI = train(40000,N,Nc,1.0/t,loss_array,lr=5e-4)
#   entropy.append(compute_entropy(temp_PCI.detach().numpy()))
 
  
# plt.figure(figsize=(8, 6))
# X = np.arange(75,90,1)
# plt.plot(X, entropy) 
# plt.xticks(X)
# plt.show()

# %%
labels = np.argmax(P_C_i,axis=1)

# %% [markdown]
# #### Iterative Approach

# %%
import numpy as np

mutual_info = np.loadtxt("temp/normalized_ticket_sij.csv", delimiter=",")

# %%
def compute_P_C(P, N):
    return np.sum(P,axis=0)/N

def compute_s_C(P, similarity_matrix, C, N):
    P_C = compute_P_C(P, N)
    P_column = P[:, C]
    norm_factor = (P_C[C] * N) ** 2
    s_C = np.sum(similarity_matrix * np.outer(P_column, P_column)) / norm_factor
    return s_C
    
def compute_s_C_i(P, similarity_matrix, C, i, N):
    P_C = compute_P_C(P, N)
    s_C_i = 0
    for j in range(N):
        s_C_i += similarity_matrix[i, j]*P[j, C]
    s_C_i /= (P_C[C]*N)
    return s_C_i

def soft_clustering(similarity_matrix, T, Nc, epsilon):
    N = similarity_matrix.shape[0]
    np.random.seed(42)
    P = np.random.rand(N, Nc)
    P = P / P.sum(axis=1, keepdims=True)  # Normalize distribution
    count = 0
    P_new = np.zeros_like(P)
    
    while True:
        P_C = compute_P_C(P,N)
        for C in range(Nc):
          s_C = compute_s_C(P, similarity_matrix, C, N)
          s_C_i = np.zeros(N)
          for i in range(N):
              s_C_i[i] = compute_s_C_i(P, similarity_matrix, C, i, N)  
          exponent = (1 / T) * (2 * s_C_i - s_C)
          P_new[:,C] = P_C[C] * np.exp(exponent)
        # Normalize P_new
        P_new = P_new / P_new.sum(axis=1, keepdims=True)

        # Check convergence
        if np.all(np.abs(P_new - P) <= epsilon):
            break
        
        count += 1
        print(f"Epoch {count}, diff is {np.sum(np.abs(P_new - P))}")
        
        P = P_new.copy()

    return P

# %%
T = 1.0/60
Nc = 11
N = 496
epsilon = 1e-5

P_C_i = soft_clustering(mutual_info,T,Nc,epsilon)

np.savetxt("temp/ticket_opt_P_C_i.csv",P_C_i,delimiter=",")

# %%
P_C_i = np.loadtxt("temp/ticket_opt_P_C_i.csv",delimiter=",")

def draw(P_C_i,i):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, len(P_C_i[i]), 1)
    plt.bar(x, P_C_i[i], width=0.8, color='skyblue', align='center')
    plt.xticks(x)
    plt.show()
    print(np.max(P_C_i[i]))
    
draw(P_C_i,0)

# %%
N=496
Nc = 11
P_C = compute_P_C(P_C_i,N)
s_C_array = []
for i in range(Nc):
  s_C = compute_s_C(P_C_i,mutual_info,i,N)
  s_C_array.append(s_C)
s_C = np.array(s_C_array)
F = np.sum(P_C*s_C) - T*np.sum(P_C_i * np.log2(P_C_i / P_C)) / N

print(F)
print(P_C)

# %%
entropy = []

def compute_entropy(P_C_i):
    return -np.sum(P_C_i * np.log2(P_C_i)) if np.all(P_C_i > 0) else 0
  
for t in range(40,80):
  P_C_i = soft_clustering(mutual_info,1/t,Nc,1e-4)
  draw(P_C_i,0)
  entropy.append(compute_entropy(P_C_i))

plt.figure(figsize=(8, 6))
X = np.arange(40,80,1)
plt.plot(X, entropy) 
plt.xticks(X)
plt.show()

# %%
P_C_i = soft_clustering(mutual_info,1.0/60,Nc,epsilon)

labels = np.argmax(P_C_i,axis=1)

# %%
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.metrics import accuracy_score


y_encoded = LabelEncoder().fit_transform(category.to_numpy())
y_pred = np.argmax(P_C_i, axis=1)

labels = np.zeros_like(y_pred)
for i in range(Nc):
    mask = (y_pred == i)
    labels[mask] = mode(y_encoded[mask],keepdims=True)[0][0]

accuracy = accuracy_score(y_encoded, labels)

print(f"Accuracy:{accuracy:.4f}")

# %% [markdown]
# It can be found that the labels calculated by different method are the same.

# %% [markdown]
# | Nc  | k-means (Mutual Info) | Info-based (Mutual Info) | k-means (Pearson) | Info-based (Pearson) |
# |-----|------------------------|--------------------------|-------------------|-----------------------|
# | 20  | 0.157                 | 0.127                   | 0.384            | 0.354                |
# | 10  | 0.131                 | 0.110                   | 0.320            | 0.307                |
# | 5   | 0.113                 | 0.110                   | 0.264            | 0.276                |
# 

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
Nc = [20, 10, 5]
k_means_mutual_info = [0.157, 0.131, 0.113]
info_based_mutual_info = [0.127, 0.110, 0.110]

# Plot with four distinct columns for clarity
plt.figure(figsize=(10, 6))

# Adjust bar positions
bar_width = 0.2
x_indices = np.arange(len(Nc))

# Mutual Information bars
plt.bar(x_indices - 1.5 * bar_width, k_means_mutual_info, bar_width, label='k-means (Mutual Info)', color='skyblue')
plt.bar(x_indices - 0.5 * bar_width, info_based_mutual_info, bar_width, label='Info-based (Mutual Info)', color='dodgerblue')

# Labels and title
plt.xlabel('Nc')
plt.ylabel('<s>')
plt.xticks(x_indices, Nc)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Testing

# %%
from sklearn.metrics import pairwise_distances
 
pearson_similarity_matrix = 1 - pairwise_distances(change_ratio_array, metric='correlation')

print(pearson_similarity_matrix)
plt.figure(figsize=(10, 8))
plt.imshow(pearson_similarity_matrix)
plt.colorbar()
plt.show()

# %%
T = 1.0/80
Nc = 11
N = 150
epsilon = 1e-4

P_C_i = soft_clustering(pearson_similarity_matrix,T,Nc,epsilon)

# %%
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.metrics import accuracy_score


y_encoded = LabelEncoder().fit_transform(category.to_numpy())

y_pred = np.argmax(P_C_i, axis=1)

labels = np.zeros_like(y_pred)
for i in range(Nc):
    mask = (y_pred == i)
    labels[mask] = mode(y_encoded[mask])[0][0]

accuracy = accuracy_score(y_encoded, labels)

print(f"Accuracy:{accuracy:.4f}")

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=11, random_state=42)
kmeans.fit(change_ratio_array)

y_pred = kmeans.labels_
labels = np.zeros_like(y_pred)
for i in range(Nc):
    mask = (y_pred == i)
    labels[mask] = mode(y_encoded[mask],keepdims=True)[0][0]

accuracy = accuracy_score(y_encoded, labels)

print(f"Kmeans Accuracy:{accuracy:.4f}")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



