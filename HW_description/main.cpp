#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "helper.h"

using namespace std;
using PVV = pair<vector<int>, vector<int>>;


class Preprocessor {
public:
    Preprocessor(vector<vector<float>> &data) : df(data) {}

    vector<vector<float>> preprocess() {
        _preprocess_categorical();
        _preprocess_numerical();
        _preprocess_ordinal();
        return df;
    }
private:
    vector<vector<float>> df;

    void _preprocess_categorical() {   
        //  Custom logic for preprocessing numerical features goes here
    }

    void _preprocess_numerical() {
        // Add custom logic here for categorical features
    }

    void _preprocess_ordinal() {
        // Custom logic for preprocessing ordinal features goes here
    }
};


class Classifier {
public:
    virtual void fit(vector<vector<float>> &X, vector<vector<float>> &y) = 0;
    virtual vector<vector<float>> predict(vector<vector<float>> &X) = 0;
    virtual vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) = 0;
};

class NaiveBayesClassifier: public Classifier {
public:
    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        // Implement the fitting logic for Naive Bayes classifier
        
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for Naive Bayes classifier
        return vector<vector<float>>();
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for Naive Bayes classifier
        return vector<unordered_map<int, float>>();
    }

private: 
    // Implement private function or variable if you needed

};


class KNearestNeighbors: public Classifier {
public:
    KNearestNeighbors(int k = 3): k(k) {} // constructor

    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        // Implement the fitting logic for KNN
        
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for KNN
        return vector<vector<float>>();
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for KNN
        return vector<unordered_map<int, float>>();
    }

private: 
    // Implement private function or variable if you needed
    int k;
};

class MultilayerPerceptron: public Classifier {
public:
    MultilayerPerceptron(int input_size=64, int hidden_size=64, int output_size=64)
                        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        // Initialize the neural network parameters (e.g., weight matrices and bias vectors)
    }
    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        // Implement training logic for MLP including forward and backward propagation
        
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement prediction logic for MLP
        return vector<vector<float>>();
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for MLP
        return vector<unordered_map<int, float>>();
    }
    void _forward_propagation(vector<vector<float>> &X) {
        // Implement forward propagation for MLP
    }
    void _backward_propagation(vector<vector<float>> &output, vector<vector<float>> &target) {
        // Implement backwardpropagation for MLP

    }


private: 
    // Implement private function or variable if you needed
    int input_size;
    int hidden_size;
    int output_size;
    int epochs;
    float learning_rate;

};


unordered_map<string, float> evaluate_model(Classifier*, vector<vector<float>>&, vector<vector<float>>&, float);

int main() {
    string train_pth = "trainWithLabel.csv";
    string test_pth = "testWithoutLabel.csv";
    vector<vector<float>> train_df = read_csv_file(train_pth);
    vector<vector<float>> test_df = read_csv_file(test_pth);
    /*create dictionary of models for iterating*/
    unordered_map<float, Classifier*> models;
    models[1.0f] = new NaiveBayesClassifier(); // 1.0 represents Naive Bayes
    models[2.0f] = new KNearestNeighbors(); // 2.0 represents KNN 
    models[3.0f] = new MultilayerPerceptron(); // 3.0 represents MLP 
    
    /*preprocessing*/
    Preprocessor train_preprocessor(train_df), test_preprocessor(test_df);
    train_df = train_preprocessor.preprocess();
    test_df = test_preprocessor.preprocess();

    /*split the dataset*/
    vector<vector<float>> X_train = get_X(train_df), y_train = get_y(train_df);
    vector<vector<float>> X_test = get_X_test(test_df);

    /*k fold cross-validation*/
    int n_splits = 10, random_state = 42;
    vector<PVV> folds = k_fold_split(X_train, y_train, n_splits, random_state);
    vector<vector<unordered_map<string, float>>> cv_result;
    
    for (auto &p: models) {
        Classifier* model = p.second;
        float model_label = p.first;
        vector<unordered_map<string, float>> fold_result;

        for (int fold = 0; fold < folds.size(); fold++) {
            auto &train_indices = folds[fold].first;
            auto &val_indices = folds[fold].second;
            // get X_fold, y_fold
            vector<vector<float>> X_train_fold, y_train_fold, X_val_fold, y_val_fold;
            
            for (auto &idx: train_indices) {
                X_train_fold.push_back(X_train[idx]);
                y_train_fold.push_back(y_train[idx]);
            }
            for (auto &idx: val_indices) {
                X_val_fold.push_back(X_train[idx]);
                y_val_fold.push_back(y_train[idx]);
            }
            model->fit(X_train_fold, y_train_fold);
            unordered_map<string, float> res = evaluate_model(model, X_val_fold, y_val_fold, model_label);
            fold_result.push_back(res);
        }
        cv_result.push_back(fold_result);
    }
    write_result_to_csv(cv_result);

    unordered_map<float, vector<vector<float>>> predictions;

    for (auto &p: models) {
        Classifier* model = p.second;
        predictions[p.first] = model->predict(X_test);
    }
    write_predictions_to_csv(predictions);
    cout << "Model predictions saved to test_results.csv\n";

    
    return 0;
}

unordered_map<string, float> evaluate_model(Classifier* model, vector<vector<float>>& X_test, vector<vector<float>>& y_test, float model_label) {
    vector<vector<float>> predictions = model->predict(X_test);
    vector<unordered_map<int, float>> proba_array = model->predict_proba(X_test);

    float accuracy = accuracy_score(y_test, predictions);
    float precision = precision_score(y_test, predictions);  // positive label = 1
    float recall = recall_score(y_test, predictions);
    float f1 = f1_score(recall, precision);
    float mcc = matthews_corrcoef(y_test, predictions);
    double auc = roc_auc_score(y_test, proba_array);

    return {{"model", model_label}, {"accuracy", accuracy}, {"f1", f1}, {"precision", precision},
            {"recall", recall}, {"mcc", mcc}, {"auc", auc}};
}
