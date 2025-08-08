import numpy as np 
import matplotlib.pyplot as mplt
# for best fit 
from scipy.optimize import curve_fit
#make a big for loop for N 

eigenvector_orthogonality = dict()

# mapping: matrix -> [dot_1_2, dot_4_8, ...]

for N in range (100, 1000, 50):
# Creating S with r = 10 where S is a square diagonal matrix
#matrix_1 = numpy.
#a_ii = rng.standard_normal(1)
#matrix_S = np.zeros((10,10))
#matrix_S = [[5 6 6.5 6.5 5.5 7 7 8 9 6.5][8 7 6 5.5 5 6.3 7 8.3 9.1 6][6 7.5 8 9 6.5 6.5 5.3 5.4 6.2 9 5.8]
 #           [8.2 7 5 6.5 7 5.3 8 6.7 7.1 5][6.3 9 8.1 7 9 6.4 8.2 7.9 9.1 5.6][5 6.3 7 9.2 8.8 7.3 9 8 5.8 6.3]
  #          [9 8.9 7.1 6 5 6 5.5 6.7 5.2 7][5.7 9 8.7 7.3 6.2 9 5.5 7.9 5 6.1][5.7 9.2 8.8 7.4 8 9.1 5.6 7.2 6.9 9.2]
   #         [5.2 9 8 6 7 9.3 8.9 5.1 7 6.1]]

   # initialize mapping for N
   eigenvector_orthogonality[N] = []

   matrix_S = np.zeros((N, N))
   matrix_S[1][1] = 10
   matrix_S[2][2] = 9.5
   matrix_S[3][3] = 8.7
   matrix_S[4][4] = 8.2
   matrix_S[5][5] = 8
   matrix_S[6][6] = 7.6
   matrix_S[7][7] = 7
   matrix_S[8][8] = 6.4
   matrix_S[9][9] = 5.8
   matrix_S[10][10] = 5.2

#matrix_S = np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #[0, 9.5, 0, 0, 0, 0, 0, 0, 0, 0],
            #[0, 0, 8.7, 0, 0, 0, 0, 0, 0, 0],
            #[0, 0, 0, 8.2, 0, 0, 0, 0, 0, 0],
            #[0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
            #[0, 0, 0, 0, 0, 7.6, 0, 0, 0, 0],
            #[0, 0, 0, 0, 0, 0, 0.03, 0, 0, 0],
            #[0, 0, 0, 0, 0, 0, 0, 0.002, 0, 0],
            #[0, 0, 0, 0, 0, 0, 0, 0, 0.0018, 0],
            #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001]]) 
# Create S st it has r (rank) eigenvalues and then depending on r we just add N-r zeroes on the diaonals 
#for i in range(10):
   #matrix_S[i][i] = np.random.standard_normal(1)
# Assign the values by hand (something like 5, 6, 7 etc.)
#matrix_S = np.round(matrix_S, 2) #rounding 

   #print(matrix_S)

# Creating R with rnadom entries with N(0, 1/N)
   matrix_R = np.zeros((N,N))

   for i in range (N):
      for j in range (N):
         matrix_R[i][j] = (1 / N) * np.random.standard_normal(1)[0]

   R_final = matrix_R + matrix_R.T - np.diag(matrix_R.diagonal()) #look for an easier way
   #print(R_final)

# Finidng W = R+S

   W = R_final + matrix_S

# SVD for W (output will be Uw, Sum and Vw)
   U_w, Sigma_w, Vt_w = np.linalg.svd(W, True, True, False)
   #print (U_w)
   #print (Sigma_w)
   #print (Vt_w)

#SVD for S just to make it faster 
   U_s, Sigma_s, Vt_s = np.linalg.svd(matrix_S, True, True, False)
   #print (U_s)
   #print (Sigma_s)
   #print (Vt_s)
# Uw = Uw' => to compare W' and S we take the <u_tilda_i, u_i> where u_tilda_i is in Uw=Uw' and u_i is in Vs which is a identity matrix bc S is diagonal
# Dot products of same indicies
   dot_1_1 = np.dot(U_w[:, 1], U_s[:, 1])
   squared_dot_1_1 = np.abs(dot_1_1)**2
   eigenvector_orthogonality[N].append(("squared_dot_1_1", squared_dot_1_1))

   dot_2_2 = np.dot(U_w[:, 2], U_s[:, 2])
   squared_dot_2_2 = np.abs(dot_2_2)**2
   eigenvector_orthogonality[N].append(("squared_dot_2_2", squared_dot_2_2))

   dot_3_3 = np.dot(U_w[:, 3], U_s[:, 3])
   squared_dot_3_3 = np.abs(dot_3_3)**2
   eigenvector_orthogonality[N].append(("squared_dot_3_3", squared_dot_3_3))

# Dot products of different indicies

   dot_1_2 = np.dot(U_w[:, 1], U_s[:, 2])
   squared_dot_1_2 = np.abs(dot_1_2)**2
   eigenvector_orthogonality[N].append(("squared_dot_1_2", squared_dot_1_2))


   dot_3_4 = np.dot(U_w[:, 3], U_s[:, 4])
   squared_dot_3_4 = np.abs(dot_3_4)**2
   eigenvector_orthogonality[N].append(("squared_dot_3_4", squared_dot_3_4))

   dot_5_8 = np.dot(U_w[:, 5], U_s[:, 8])
   squared_dot_5_8 = np.abs(dot_5_8)**2
   eigenvector_orthogonality[N].append(("squared_dot_5_8", squared_dot_5_8))



# start with cmparing same indicies <u_tilda_1, u_1> and fixed r: |<u_tilda_1, u_1>|^2 ---> 1 - 1/(corresp. sing. value)^2
#form each loop make a sequence of pts (|<u_tilda_1, u_1>|^2) for a couple of  SAME indicies on the graph. It should converge to the same value 1 - 1/(sigma_i)^2 (should form a line on the graph)
#form each loop make a sequence of pts (|<u_tilda_1, u_1>|^2) for a couple of  DIFERENT indicies on the graph. It should converge to zero
# calculate the rate of convergence (figure out what type of function you have) -> try finding a best fit and then comparing to 1/sqrt(n) for example
# we need the value for alfa (check your phone)
# W = Uw(sigma_w)VwT -> W' = Uw(sigma_w)' VwT

# Generate Uw and Vw
#matrix_Uw = np.zeros((10,10))
#for i in range(10):
 #   matrix_Uw[i][i] = 1
#print("Uw = ")
#print(matrix_Uw)

#matrix_Vw = np.zeros((10,10))
#for i in range(10):
#    matrix_Vw[i][i] = 1
#print("Vw = ")
#print(matrix_Vw)

# Findng Singular Value Decomposition (SVD) for W

# Take a dor product <ui, ui_tilda>

print("Orthogonality Value Mappings:")
print (eigenvector_orthogonality)
'''
Matplotlib Graphs
'''


# Figure 1 " Same indicies vs N"
# Plot 1 "|<u_tilda_1, u_1>|^2 vs N"
mplt.figure(1)
mplt.title("Same indicies d.p. vs N")
mplt.xlabel("N")
mplt.ylabel("|<u_tilda_i, u_i>|^2")

x_axis = np.arange(100, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][0][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="blue", label="|<u_tilda_i, u_i>|^2")
ref_curve_1 = np.full_like(x_axis, 0.99, dtype=np.float64)
mplt.plot(x_axis, ref_curve_1, linestyle='--', color='orange', label='1 - 1/100 (ref)')
mplt.legend()

#for best fit
x_data = np.array(x_axis)


def same_index_fit(N, a, b):
    return 1 - a / (N**b)

for idx, label in zip([0, 1, 2], ["u_tilda_1 * u_1", "u_tilda_2 * u_2", "u_tilda_3 * u_3"]):
    y_data = np.array([eigenvector_orthogonality[N][idx][1] for N in x_axis])
    popt, _ = curve_fit(same_index_fit, x_data, y_data, p0=(1, 1))
    y_fit = same_index_fit(x_data, *popt)
    
    mplt.figure(idx + 1)
    mplt.plot(x_data, y_fit, linestyle="--", label=f"Best fit: 1 - {popt[0]:.2f}/N^{popt[1]:.2f}")
    mplt.legend()


# Plot 2 "|<u_tilda_2, u_2>|^2 vs N"
#mplt.title("dot_2_2 vs N")
#mplt.xlabel("N")
#mplt.ylabel("|<u_tilda_2, u_2>|^2")
mplt.figure(2)
mplt.title("Same indicies d.p. vs N")
mplt.xlabel("N")
mplt.ylabel("|<u_tilda_2, u_2>|^2")
x_axis = np.arange(50, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][1][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="green", label="|<u_tilda_2, u_2>|^2")

ref_curve_2 = np.full_like(x_axis, 1 - 1/90.25, dtype=np.float64)
mplt.plot(x_axis, ref_curve_1, linestyle='--', color='orange', label='1 - 1/90.25 (ref)')

# Plot 3 "|<u_tilda_3, u_3>|^2 vs N"
#mplt.xlabel("N")
#mplt.ylabel("|<u_tilda_2, u_2>|^2")
mplt.figure(3)
mplt.title("Same indicies d.p. vs N")
mplt.xlabel("N")
mplt.ylabel("|<u_tilda_3, u_3>|^2")
x_axis = np.arange(50, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][2][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="red", label="|<u_tilda_2, u_2>|^2")

ref_curve_3 = np.full_like(x_axis, 1 - 1/75.69, dtype=np.float64)
mplt.plot(x_axis, ref_curve_1, linestyle='--', color='orange', label='1 - 1/75.69 (ref)')


# Figure 2 " Different indicies vs N"
# Plot 1 "|<u_tilda_1, u_2>|^2 vs N"
mplt.figure(4)
mplt.title("Different indicies d.p. vs N")
mplt.xlabel("N")
mplt.ylabel("|<u_tilda_i, u_j>|^2")

x_axis = np.arange(50, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][3][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="blue", label="|<u_tilda_1, u_2>|^2")

# Plot 2 "|<u_tilda_3, u_4>|^2 vs N"
#mplt.title("dot_2_2 vs N")
#mplt.xlabel("N")
#mplt.ylabel("|<u_tilda_2, u_2>|^2")

x_axis = np.arange(100, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][4][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="green", label="|<u_tilda_3, u_4>|^2")

# Plot 3 "|<u_tilda_5, u_8>|^2 vs N"
#mplt.xlabel("N")
#mplt.ylabel("|<u_tilda_2, u_2>|^2")
x_axis = np.arange(100, 1000, 50)
y_axis = []
for k in eigenvector_orthogonality:
   y_axis.append(eigenvector_orthogonality[k][5][1])
mplt.plot(x_axis, y_axis, marker="o", linestyle="-", color="red", label="|<u_tilda_5, u_8>|^2")

# Plot 4 - to compare 
ref_curve = 1 / (x_axis ** 2)

mplt.plot(x_axis, ref_curve, linestyle='-', color='black', label='1/N² (ref)')
mplt.legend()

def diff_index_fit(N, a, b):
    return a / (N**b)

for idx, label in zip([3, 4, 5], ["ũ₁·u₂", "ũ₃·u₄", "ũ₅·u₈"]):
    y_data = np.array([eigenvector_orthogonality[N][idx][1] for N in x_axis])
    popt, _ = curve_fit(diff_index_fit, x_data, y_data, p0=(1, 2))
    y_fit = diff_index_fit(x_data, *popt)
    
    mplt.figure(4)
    mplt.plot(x_data, y_fit, linestyle="--", label=f"Best fit ({label}): {popt[0]:.2f}/N^{popt[1]:.2f}")
    mplt.legend()

print("Orthogonality Value Mappings:")
print (eigenvector_orthogonality)


mplt.show()



def same_index_fit(N, a, b):
    return 1 - a / (N**b)

def diff_index_fit(N, a, b):
    return a / (N**b)

x_data = np.array(x_axis)

# === Same index fits ===
for idx, label in zip([0, 1, 2], ["ũ₁·u₁", "ũ₂·u₂", "ũ₃·u₃"]):
    y_data = np.array([eigenvector_orthogonality[N][idx][1] for N in x_axis])
    popt, _ = curve_fit(same_index_fit, x_data, y_data, p0=(1, 1))
    y_fit = same_index_fit(x_data, *popt)
    
    mplt.figure(idx + 1)
    mplt.plot(x_data, y_fit, linestyle="--", label=f"Best fit: 1 - {popt[0]:.2f}/N^{popt[1]:.2f}")
    mplt.legend()

# === Different index fits ===
for idx, label in zip([3, 4, 5], ["ũ₁·u₂", "ũ₃·u₄", "ũ₅·u₈"]):
    y_data = np.array([eigenvector_orthogonality[N][idx][1] for N in x_axis])
    popt, _ = curve_fit(diff_index_fit, x_data, y_data, p0=(1, 2))
    y_fit = diff_index_fit(x_data, *popt)
    
    mplt.figure(4)
    mplt.plot(x_data, y_fit, linestyle="--", label=f"Best fit ({label}): {popt[0]:.2f}/N^{popt[1]:.2f}")
    mplt.legend()

# Find best fit 
# Same indicies shouldn't converge to 1 that rapidly, they should converge to 1 - 1/(sing val)^2