import csv
import statistics


def load_housing_data():
    with open(r"Data\housing.csv") as housing_data_file:
        housing_data_object = csv.reader(housing_data_file)
        housing_data_list = list(housing_data_object)

    mean_total_bedroom_feature = []
    for i, row in enumerate(housing_data_list):
        if i == 0:
            continue
        elif row[4] == "":
            continue
        else:
            mean_total_bedroom_feature.append(float(row[4]))
    mean_value_total_bedroom_feature = statistics.mean(mean_total_bedroom_feature)

    x = []
    y = []
    for i, row in enumerate(housing_data_list):
        if i == 0:
            continue
        else:
            current_feature_values_in_row = []
            for j in range(len(row) - 2):
                if row[j] == "":
                    row[j] = mean_value_total_bedroom_feature
                current_feature_values_in_row.append(float(row[j]))
            x.append(current_feature_values_in_row)
            y.append(float(row[-2]))

    x_standardized = standardize_data(x)

    for row in x_standardized:
        row.insert(0, 1.0)

    return x_standardized, x, y


def standardize_data(x_standardized):
    feature_list = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    mean_std_dict = {"longitude": {}, "latitude": {}, "housing_median_age": {}, "total_rooms": {}, "total_bedrooms": {}, "population": {}, "households": {}, "median_income": {}}

    for i, element in enumerate(feature_list):
        all_values_of_current_feature = []
        for row in x_standardized:
            all_values_of_current_feature.append(row[i])
        mean_of_current_feature = statistics.mean(all_values_of_current_feature)
        std_of_current_feature = statistics.stdev(all_values_of_current_feature)
        mean_std_dict[element] = {"mean": mean_of_current_feature, "std": std_of_current_feature}

    for i, element in enumerate(feature_list):
        for row in x_standardized:
            row[i] = (row[i] - mean_std_dict[element]['mean']) / mean_std_dict[element]['std']
    
    return x_standardized
