from numpy import *

#calculate baseline predictor
def baseline_predictor(R) :

    #number of users
    n = R.__len__()

    #number of movies
    m = R[0].__len__()

    number_of_ratings = R.nonzero()[0].size

    #average of all ratings
    r_bar = (R.sum()+0.0)/number_of_ratings

    print "Average of all ratings : "
    print r_bar
    print
    
    #calculate b_u and b_i
    A = zeros((number_of_ratings, n+m))
    c = zeros((number_of_ratings, 1))

    index = 0
    for i in range(n) :
        for j in range(m) :
            if R[i][j] > 0 :
                A[index][i] = 1
                A[index][j+n] = 1
                c[index][0] = R[i][j]-r_bar
                index+=1

    A_T = A.T
    b = dot(dot(linalg.pinv(dot(A_T, A)), A_T), c)

    b_u = b[0:n]
    b_i = b[n:n+m]

    print "b_u(transpose) : "
    print b_u.T
    print
    print "b_i(transpose) : "
    print b_i.T
    print

    #calculate R_hat
    R_hat = r_bar * ones((n,m)) + dot(b_u, ones((1,m))) + dot(ones((n,1)), b_i.T)
    R_hat = cliffing(R_hat)
    
    print "R_hat : "
    print R_hat
    print
    
    return R_hat

#calculate nighborhood model
def nighborhood_model(R, R_hat, L) :

    #number of users
    n = R.__len__()

    #number of movies
    m = R[0].__len__()

    R_tilde = zeros((n,m))

    #calculate R_tilde
    for i in range(n) :
        for j in range(m) :
            if R[i][j] > 0 :
                R_tilde[i][j] = R[i][j] - R_hat[i][j]

    print "R_tilde : "
    print R_tilde
    print

    #calculate D(similarity matrix)
    D = zeros((m,m))
    for i in range(m) :
        for j in range(m) :
            if i==j :
                continue
            sum_i = 0 
            sum_j = 0
            prod = 0
            for k in range(n) :
                if R[k][i] != 0 and R[k][j] != 0 :
                    sum_i += pow(R_tilde[k][i], 2)
                    sum_j += pow(R_tilde[k][j], 2)
                    prod += R_tilde[k][i]*R_tilde[k][j]
            
            if prod!= 0:
                D[i][j] = prod / sqrt(sum_i) / sqrt(sum_j)

    print "D : "
    print D
    print

    #find neighbore set
    neighbor_map = {}
    for i in range(m) :
        neighbor_map[i] = max_indices(D[i], L)
        print str(i)+"th movie's neighbors : ",
        for j in neighbor_map[i] :
            print j,
        print
    print

    #calculate R_hat_N
    R_hat_N = R_hat.copy()
    for i in range(n) :
        for j in range(m) :
            numerator = 0.0
            denominator = 0.0
            for l in neighbor_map[j] :
                if R[i][l] != 0 :
                    numerator += D[j][l] * R_tilde[i][l]
                    denominator += abs(D[j][l])
            if denominator!= 0 :
                R_hat_N[i][j] += numerator / denominator

    R_hat_N = cliffing(R_hat_N)

    print "R_hat_N : "
    print R_hat_N
    print

    return R_hat_N
    

# calculate RMSE
def RMSE(R, T, R_hat) :
    
    #number of users
    n = R.__len__()

    #number of movies
    m = R[0].__len__()

    #calculate training error
    C = 0
    sum_of_squares = 0
    
    for i in range(n) :
        for j in range(m) :
            if R[i][j] > 0 :
                C+=1
                sum_of_squares += pow(R_hat[i][j]-R[i][j], 2)
                
    print "Training Error : "
    print sqrt(sum_of_squares/C)
    print

    #calculating test error
    C = 0
    sum_of_squares = 0
    
    for i in range(n) :
        for j in range(m) :
            if T[i][j] > 0 :
                C+=1
                sum_of_squares += pow(R_hat[i][j]-T[i][j], 2)
                
    print "Test Error : "
    print sqrt(sum_of_squares/C)
    print


# clip any predicted rating lower than 1 to 1 and any higher than 5 to 5
def cliffing(M) :
    
    #number of users
    n = M.__len__()

    #number of movies
    m = M[0].__len__()

    M = M.copy()
    
    for i in range(n) :
        for j in range(m) :
            M[i][j] = min(5,M[i][j])
            M[i][j] = max(1,M[i][j])

    return M

# find L max elements' indicies
def max_indices(V, L) :
    
    #number of movies
    m = V.__len__()

    V = abs(V)

    indices = []
    
    for i in range(L) :
        max_value = -1
        max_index = 0
        for j in range(m) :
            if V[j] > max_value :
                max_value = V[j]
                max_index = j
        indices.append(max_index)
        V[max_index] = -1
        
    return indices

if __name__ == '__main__':
    
    #R : Ratings except unavailables and test set
    R = array([[5, 4, 4, 0, 0],
               [0, 3, 5, 0, 4],
               [5, 2, 0, 0, 3],
               [0, 0, 3, 1, 2],
               [4, 0, 0, 4, 5],
               [0, 3, 0, 3, 5],
               [3, 0, 3, 2, 0],
               [5, 0, 4, 0, 5],
               [0, 2, 5, 4, 0],
               [0, 0, 5, 3, 4]])

    #T : Test set
    T = array([[0, 0, 0, 0, 5],
               [0, 0, 0, 3, 0],
               [0, 0, 0, 2, 0],
               [0, 2, 0, 0, 0],
               [0, 0, 5, 0, 0],
               [5, 0, 0, 0, 0],
               [0, 2, 0, 0, 0],
               [0, 3, 0, 0, 0],
               [4, 0, 0, 0, 0],
               [5, 0, 0, 0, 0]])

    #L : size of neighbor set
    L = 2

    #calulate baseline predictor
    R_hat = baseline_predictor(R)

    #test baseline predictor
    RMSE(R, T, R_hat)

    #calculate neighborhood model
    R_hat_N = nighborhood_model(R, R_hat, L)

    #test neighborhood model
    RMSE(R, T, R_hat_N)
