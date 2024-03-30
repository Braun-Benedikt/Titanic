from sklearn.impute import SimpleImputer

def clean_data(train_data):
    imputer = SimpleImputer(strategy='mean')
    train_data['age'] = imputer.fit_transform(train_data[['age']])
    return train_data
