# Import accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix

# # Function to predict and evaluate model 
def predict_model(model, X_test, y_test):
    # Predict based on the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    # Show confusion matriz  
    confusion_mat = confusion_matrix(y_test,y_pred)

    return accuracy,confusion_mat,y_pred
