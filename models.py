

def calculate_predictions_of_hypothesis(x, vector_theta):
    y_hat = []
    for x_i in x:
        interim_results = []
        for feature_i, theta_i in zip(x_i, vector_theta):
            interim_results.append(feature_i * theta_i)
        h_of_x = sum(interim_results)
        y_hat.append(h_of_x)

    return y_hat, vector_theta


def calculate_value_of_loss_function(y_hat, y, m):
    J = 0
    for y_i, y_hat_i in zip(y, y_hat):
        J += (y_hat_i - y_i) ** 2
    J = J / (2 * m)

    return J


def derivative_cost_function(x_standardized, y, y_hat, m, n):
    gradient = [0.0 for _ in range(n)]
    for i in range(m):
        error = y_hat[i] - y[i]
        for j in range(n):
            gradient[j] += error * x_standardized[i][j]

    gradient = [(g_i / m) for g_i in gradient]

    return gradient


def update_parameters(vector_theta, gradient, alpha):
    for i in range(len(vector_theta)):
        vector_theta[i] = vector_theta[i] - alpha * gradient[i]

    return vector_theta
