import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

linear_regression_model = RandomForestRegressor(n_estimators=100, random_state=76)
vectorizer = TfidfVectorizer()
training_data = pd.read_csv('dataset.csv')

training_data['Engine'] = training_data['Engine'].str.extract(r'(\d+\.?\d*)').astype(float)
training_data['Power'] = training_data['Power'].str.extract(r'(\d+\.?\d*)').astype(float)
training_data['Mileage'] = training_data['Mileage'].str.replace(' km/kg', '').str.replace(' kmpl', '').astype(float)
training_data['New_Price'] = training_data['New_Price'].str.replace(' Lakh', '').str.replace(' Cr', '').replace(
    'NaN', 0).astype(float)
training_data['Age'] = 2024 - training_data['Year']
predictive_column = training_data['Price']

training_data['categorical_features'] = training_data.apply(
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

x_train_numerical, x_test_numerical, x_train_categorical, x_test_categorical, y_train, y_test = train_test_split(
    training_data[numerical_features],
    training_data['categorical_features'],
    predictive_column,
    test_size=0.03,
    random_state=23
)

x_train_categorical = vectorizer.fit_transform(x_train_categorical)
x_test_categorical = vectorizer.transform(x_test_categorical)

x_train = pd.concat([pd.DataFrame(x_train_categorical.toarray()), x_train_numerical.reset_index(drop=True)], axis=1)

x_train.columns = x_train.columns.astype(str)

x_test = pd.concat([pd.DataFrame(x_test_categorical.toarray()), x_test_numerical.reset_index(drop=True)], axis=1)

x_test.columns = x_test.columns.astype(str)

linear_regression_model.fit(x_train, y_train)

joblib.dump(linear_regression_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
