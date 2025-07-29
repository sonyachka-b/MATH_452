import numpy as np 


# Creating S with r = 10 where S is a square diagonal matrix
#matrix_1 = numpy.
#a_ii = rng.standard_normal(1)
matrix_S = np.zeros((10,10))

for i in range(10):
    matrix_S[i][i] = np.random.standard_normal(1)

matrix_S = np.round(matrix_S, 2)
print(matrix_S)
# Creating R with rnadom entries with N(0, 1/N)
matrix_R = np.zeros((10,10))

for i in range (10):
    for j in range (10):
        matrix_R[i][j] = np.random.standard_normal(1)

R_final = matrix_R + matrix_R.T - np.diag(matrix_R.diagonal())
R_final = np.round(R_final, 2)
print(R_final)

# Finidng W = R+S

W = R_final + matrix_S

# Generate Uw and Vw

# Findng Singular Value Decomposition (SVD) for W

# Take a dor product <ui, ui_tilda>



