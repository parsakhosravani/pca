import numpy as np

def pca(X, num_components):
    # Subtract the mean from the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    
    # Project the data onto the selected eigenvectors
    transformed_X = np.dot(X_centered, selected_eigenvectors)
    
    return transformed_X

# Example usage
# Generate a random dataset with 100 samples and 5 features
np.random.seed(0)
X = np.random.rand(100, 5)

# Apply PCA to reduce the dimensionality to 2 components
transformed_X = pca(X, num_components=2)

print(transformed_X.shape)
print(transformed_X[:5])
