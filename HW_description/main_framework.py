from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def preprocess(self):
        # Apply various preprocessing methods on the DataFrame
        self.df = self._preprocess_numerical(self.df)
        self.df = self._preprocess_categorical(self.df)
        self.df = self._preprocess_ordinal(self.df)
        return self.df

    def _preprocess_numerical(self, df):
        # Custom logic for preprocessing numerical features goes here
        return df

    def _preprocess_categorical(self, df):
        # Add custom logic here for categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df

    def _preprocess_ordinal(self, df):
        # Custom logic for preprocessing ordinal features goes here
        return df

# Implementing the classifiers (NaiveBayesClassifier, KNearestNeighbors, MultilayerPerceptron)

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        # Initialize the classifier
        pass

    def fit(self, X, y):
        # Implement the fitting logic for Naive Bayes classifier
        pass

    def predict(self, X):
        # Implement the prediction logic for Naive Bayes classifier
        pass
    
    def predict_proba(self, X):
        # Implement probability estimation for Naive Bayes classifier
        pass

# K-Nearest Neighbors Classifier
class KNearestNeighbors(Classifier):
    def __init__(self, k=3):
        # Initialize KNN with k neighbors
        self.k = k

    def fit(self, X, y):
        # Store training data and labels for KNN
        pass

    def predict(self, X):
        # Implement the prediction logic for KNN
        pass
    
    def predict_proba(self, X):
        # Implement probability estimation for KNN
        pass

# Multilayer Perceptron Classifier
class MultilayerPerceptron(Classifier):
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        # Initialize MLP with given network structure
        pass

    def fit(self, X, y, epochs, learning_rate):
        # Implement training logic for MLP including forward and backward propagation
        pass

    def predict(self, X):
        # Implement prediction logic for MLP
        pass

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        pass
        
    def _forward_propagation(self, X):
        # Implement forward propagation for MLP
        pass

    def _backward_propagation(self, output, target):
        # Implement backward propagation for MLP
        pass 

# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, proba[:, 1])
        else:  # Multiclass classification
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
    else:
        auc = None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
    
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('trainWithLabel.csv')
    
    # Preprocess the training data
    preprocessor = Preprocessor(df)
    df_processed = preprocessor.preprocess()

    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier(), 
              'KNN': KNearestNeighbors(), 
              'MLP': MultilayerPerceptron()
    }

    # Split the dataset into features and target variable
    X_train = df_processed.drop('Outcome', axis=1)
    y_train = df_processed['Outcome']
    
    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []

    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)
            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_excel('cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")

    # Load the test dataset, assuming you have a test set CSV file without labels
    df_ = pd.read_csv('testWithoutLabel.csv')
    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")

if __name__ == "__main__":
    main()
