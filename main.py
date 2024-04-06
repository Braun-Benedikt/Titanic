from data_cleaning import clean_data
from feature_engineering import feature_engineering
from model_training import evaluate_model, grid_search_logistic_regression, grid_search_random_forest
from sklearn.model_selection import train_test_split
import seaborn as sns


# Load Dataset
titanic_df = sns.load_dataset('titanic')

# Clean data
train_data = clean_data(titanic_df)

# Feature engineering
train_data = feature_engineering(train_data)

# Training and evaluating
X_train, X_val, y_train, y_val = train_test_split(train_data.drop('survived', axis=1), train_data['survived'], test_size=0.2)
logistic_regression_model = grid_search_logistic_regression(X_train, y_train)
random_forest_model = grid_search_random_forest(X_train, y_train)
evaluate_model(logistic_regression_model, X_val, y_val)
evaluate_model(random_forest_model, X_val, y_val)
