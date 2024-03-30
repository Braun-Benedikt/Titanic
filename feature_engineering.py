from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def feature_engineering(train_data):
    relevant_features = ['pclass', 'sex', 'age', 'family_size', 'fare', 'survived']
    train_data['family_size'] = train_data['sibsp'] + train_data['parch'] + 1
    train_data = train_data[relevant_features]
    # Encoding
    label_encoder = LabelEncoder()
    train_data['sex'] = label_encoder.fit_transform(train_data['sex'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data.drop('survived', axis=1))
    scaled_data = pd.DataFrame(scaled_data, columns=train_data.drop('survived', axis=1).columns)
    scaled_data['survived'] = train_data['survived']
    return scaled_data
