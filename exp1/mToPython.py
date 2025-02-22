import numpy as np
import matplotlib.pyplot as plt

def CH2_2(patterns, targets, c1, c2, P1, P2, dim):
    # Generate classification model of two classes, then calculate the classification
    # error and the Bhattacharyya bound.
    
    # Generate classification model
    x1 = patterns[:dim, targets == c1]
    x2 = patterns[:dim, targets == c2]
    
    u1 = np.mean(x1, axis=1).reshape(-1, 1)
    u2 = np.mean(x2, axis=1).reshape(-1, 1)
    
    sigma1 = np.atleast_2d(np.cov(x1))
    sigma2 = np.atleast_2d(np.cov(x2))
    
    model = {
        'u1': u1, 'u2': u2,
        'sigma1': sigma1, 'sigma2': sigma2,
        'P1': P1, 'P2': P2
    }
    
    # Calculate the classification error
    def classify(x, u, sigma, P):
        # Multivariate normal distribution
        d = x.shape[0]
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm = 1 / ((2*np.pi)**(d/2) * np.sqrt(det))
        exp = np.exp(-0.5 * np.dot(np.dot((x-u).T, inv), (x-u)))
        return np.log(norm * exp * P)
    
    result1 = np.array([c1 if classify(x.reshape(-1,1), u1, sigma1, P1) >= 
                        classify(x.reshape(-1,1), u2, sigma2, P2) else c2 for x in x1.T])
    
    result2 = np.array([c2 if classify(x.reshape(-1,1), u2, sigma2, P2) >= 
                        classify(x.reshape(-1,1), u1, sigma1, P1) else c1 for x in x2.T])
    
    error_num = np.sum(result1 == c2) + np.sum(result2 == c1)
    error = error_num / (x1.shape[1] + x2.shape[1])
    
    # Bhattacharyya bound
    def Bhattacharyya(u1, sigma1, u2, sigma2, P1):
        P2 = 1 - P1
        sigma = (sigma1 + sigma2) / 2
        term1 = 1/8 * np.dot(np.dot((u1-u2).T, np.linalg.inv(sigma)), (u1-u2))
        term2 = 1/2 * np.log(np.linalg.det(sigma) / np.sqrt(np.linalg.det(sigma1) * np.linalg.det(sigma2)))
        return np.sqrt(P1*P2) * np.exp(-term1 - term2)
    
    Bbound = Bhattacharyya(u1, sigma1, u2, sigma2, P1)
    
    return model, error, Bbound

# Hardcode the data
data = np.array([
    [-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03],
    [-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33],
    [-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33],
    [-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50],
    [-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32],
    [-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31],
    [5.35, 5.12, -1.34, 4.48, 7.11, 7.17, 5.75, 0.77, 0.90, 3.52],
    [2.26, 3.22, -5.31, 3.42, 2.39, 4.33, 3.97, 0.27, -0.43, -0.36],
    [8.13, -2.66, -9.87, 5.19, 9.21, -0.98, 6.65, 2.41, -8.71, 6.43]
])

# Transpose the data to have features as rows and samples as columns
patterns = data

# Create target labels
targets = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

error_vec = []
Bbound_vec = []
model_list = []

# Calculate classification error and Bhattacharyya bound
for dim in range(1, 4):
    model, error, Bbound = CH2_2(patterns, targets, 1, 2, 0.5, 0.5, dim)
    model_list.append(model)
    error_vec.append(error)
    Bbound_vec.append(Bbound)

# Plot
plt.figure()
plt.plot(range(1, 4), error_vec, '--ro', label='classification error')
plt.plot(range(1, 4), Bbound_vec, '-g*', label='Bhattacharyya bound')
plt.xlabel('dim')
plt.ylabel('error')
plt.legend()
plt.show()

# Print results
for dim, error, Bbound in zip(range(1, 4), error_vec, Bbound_vec):
    print(f"Dimension: {dim}")
    print(f"Classification Error: {error:.4f}")
    print(f"Bhattacharyya Bound: {Bbound:.4f}")
    print()