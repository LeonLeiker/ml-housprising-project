import matplotlib.pyplot as plt


def plot_and_save_training_data(x, y, y_hat):
    plt.figure(figsize=(8, 5))
    plt.scatter(x=x, y=y, color='blue', s=10)
    plt.plot(x, y_hat, color='red', linewidth=2)
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.title('California Housing Data - Training Set')
    plt.savefig(r'C:\Einsteiger Projekte\Linear Regression\Results\training_data_plot.png', dpi=300)


def plot_and_save_loss_function(J_sqrt):
    epochs = [epoch for epoch, J_sqrt_value in J_sqrt]
    J_sqrt_values = [J_sqrt_value for epoch, J_sqrt_value in J_sqrt]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, J_sqrt_values, color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Progression of Loss Function (RMSE)')
    plt.savefig(r'C:\Einsteiger Projekte\Linear Regression\Results\loss_function_progression.png', dpi=300)
