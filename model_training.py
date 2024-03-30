from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def grid_search_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 30, 50, 80, 100]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    return grid_search.best_estimator_

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    val_score = model.score(X_val, y_val)
    print("Validation accuracy:", val_score)
