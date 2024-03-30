from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

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
    y_predicted = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_predicted))
    # Calculating precision-recall values
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    # Plotting precision-recall curve
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    # Calculating ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()