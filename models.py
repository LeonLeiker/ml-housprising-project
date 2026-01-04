

def calculate_predictions_of_hypothesis(x, theta_0, theta_1):
    y_hat = []
    for x_i in x:
        h_of_x = theta_1 * x_i + theta_0
        y_hat.append(h_of_x)

    return y_hat


def calculate_value_of_loss_function(y_hat, y, m):
    J = 0
    for y_1, y_hat_1 in zip(y, y_hat):
        J += (y_hat_1 - y_1) ** 2
    J = J / (2 * m)

    return J


def derivative_cost_function(y_hat, y, x, m):
    d_theta_0 = 0
    d_theta_1 = 0

    for y_hat_1, y_1, x_1 in zip(y_hat, y, x):
        d_theta_0 += (y_hat_1 - y_1)
        d_theta_1 += (y_hat_1 - y_1) * x_1

    d_theta_0 = d_theta_0 / m
    d_theta_1 = d_theta_1 / m

    return d_theta_0, d_theta_1


def update_parameters(theta_0, theta_1, d_theta_0, d_theta_1, alpha):
    theta_0 = theta_0 - alpha * d_theta_0
    theta_1 = theta_1 - alpha * d_theta_1

    return theta_0, theta_1
