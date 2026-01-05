import statistics
from data import load_housing_data
from plots import plot_and_save_training_data, plot_and_save_loss_function
from models import calculate_predictions_of_hypothesis, calculate_value_of_loss_function, derivative_cost_function, update_parameters


def main():
    x_standardized, x, y = load_housing_data()
    J_sqrt = []
    m = len(y) # number of training examples
    n = len(x_standardized[0]) # number of features
    vector_theta = [0.0 for _ in range(n)]
    alpha = 0.01
    training_epochs = 2000

    for i in range(training_epochs):
        y_hat, vector_theta = calculate_predictions_of_hypothesis(x_standardized, vector_theta)
        J = calculate_value_of_loss_function(y_hat, y, m)
        J_sqrt.append((i, statistics.sqrt(2 * J)))
        if i % 10 == 0:
            print(f"Epoch {i}: Loss Function Value = {J}")
        gradient = derivative_cost_function(x_standardized, y, y_hat, m, n)
        vector_theta = update_parameters(vector_theta, gradient, alpha)

    plot_and_save_loss_function(J_sqrt)
    plot_and_save_training_data(x, y, y_hat)
   

if __name__ == "__main__":
    main()
