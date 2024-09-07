from joblib import load
import pandas as pd


def prediction(file_path):
    linear_regression_model = load('random_forest_model.pkl')
    vectorizer = load('vectorizer.pkl')

    test_data = pd.read_csv(file_path)

    test_data['Engine'] = test_data['Engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    test_data['Power'] = test_data['Power'].str.extract(r'(\d+\.?\d*)').astype(float)
    test_data['Mileage'] = test_data['Mileage'].str.replace(' km/kg', '').str.replace(' kmpl', '').astype(float)
    test_data['New_Price'] = test_data['New_Price'].str.replace(' Lakh', '').str.replace(' Cr', '').replace('NaN',
                                                                                                            0).astype(
        float)
    test_data['Age'] = 2024 - test_data['Year']

    test_data['combined_features'] = test_data.apply(
        lambda row: ' '.join([
            str(row['Name']),
            str(row['Location']),
            str(row['Fuel_Type']),
            str(row['Transmission']),
            str(row['Owner_Type'])
        ]),
        axis=1
    )
    numerical_features = ['Engine', 'Power', 'Mileage', 'New_Price', 'Age']
    x_test_numerical = test_data[numerical_features]
    
    x_test_categorical = vectorizer.transform(test_data['combined_features'])

    x_test = pd.concat([pd.DataFrame(x_test_categorical.toarray()), x_test_numerical.reset_index(drop=True)],
                       axis=1)

    x_test.columns = x_test.columns.astype(str)

    predictions = linear_regression_model.predict(x_test)

    test_data['Predicted_Price'] = predictions

    test_data.to_csv('car_price_predictions.csv', index=False)

    return 'car_price_predictions.csv'
