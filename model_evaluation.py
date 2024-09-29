from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("Feature Importances:")
    print(model.feature_importances_)
