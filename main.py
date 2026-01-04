import statistics
from data import load_housing_data
from plots import plot_and_save_training_data, plot_and_save_loss_function
from models import calculate_predictions_of_hypothesis, calculate_value_of_loss_function, derivative_cost_function, update_parameters


def main():
    x_standardized, x, y = load_housing_data()
    J_sqrt = []
    m = len(y)
    theta_0 = 0
    theta_1 = 0
    alpha = 0.001
    training_epochs = 3000

    for i in range(training_epochs):
        y_hat = calculate_predictions_of_hypothesis(x_standardized, theta_0, theta_1)
        J = calculate_value_of_loss_function(y_hat, y, m)
        J_sqrt.append((i, statistics.sqrt(2 * J)))
        if i % 10 == 0:
            print(f"Epoch {i}: Loss Function Value = {J}")
            print(f"Theta 0: {theta_0}, Theta 1: {theta_1}")
        d_theta_0, d_theta_1 = derivative_cost_function(y_hat, y, x_standardized, m)
        theta_0, theta_1 = update_parameters(theta_0, theta_1, d_theta_0, d_theta_1, alpha)

    plot_and_save_loss_function(J_sqrt)
    plot_and_save_training_data(x, y, y_hat)
   

if __name__ == "__main__":
    main()
