from sklearn.preprocessing import LabelEncoder
def feature_engineering(train_data):
    train_data['family_size'] = train_data['sibsp'] + train_data['parch'] + 1
    # Encoding
    label_encoder = LabelEncoder()
    train_data['sex'] = label_encoder.fit_transform(train_data['sex'])
    return train_data
