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
#include <random>
#include "helper.h"

using namespace std;
using PVV = pair<vector<int>, vector<int>>;

float max0[17],min0[17];
float mean0[17];
float most0[60];
bool validation = false;

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
        for (int i = 17; i < 77; ++i) {
            std::vector<float> column = get_column(i);
            if(!validation){                                           
                int zerocount = 0;
                int onecount = 0;
                for (float& val : column) {
                    if(val == 0){
                        zerocount++;
                    }
                    if(val == 1){
                        onecount++;
                    }
                }
                for (float& val : column) {
                    if(val == -1){
                        val = (onecount>=zerocount)? 1 : 0 ;
                    } 
                }
                most0[i-17] = (onecount>=zerocount)? 1 : 0 ;
            }else{
                for (float& val : column) {
                    if(val == -1){
                        val = most0[i-17];
                    } 
                }
            }
            set_column(i, column);
        }
    }

    void _preprocess_numerical() {
        for (int i = 0; i < 17; ++i) {
            std::vector<float> column = get_column(i);
            if(!validation){
                float mean = calculate_mean(column);
                for (float& val : column) {
                    if(val == -1){
                        val = mean;
                    }
                }
                mean0[i] = mean;
                vector<float> temp;
                for(int i=0;i<column.size();i++){
                    temp.push_back(column[i]);
                }
                std::sort(temp.begin(),temp.end());
                min0[i] = temp[0];
                max0[i] = temp[temp.size()-1];
                for (float& val : column) {
                    val = (val-temp[0])/(temp[temp.size()-1]-temp[0]);
                }
            }
            else{
                for (float& val : column) {
                    if(val == -1){
                        val = mean0[i];
                    }
                }
                for (float& val : column) {
                    val = (val-min0[i])/(max0[i]-min0[i]);
                }
            }
            
            set_column(i, column);
        }
    }

    void _preprocess_ordinal() {
        // Custom logic for preprocessing ordinal features goes here
    }

    vector<float> get_column(int index) {
        vector<float> column(df.size());
        for (size_t i = 0; i < df.size(); ++i) {
            column[i] = df[i][index];
        }
        return column;
    }

    void set_column(int index, const std::vector<float>& column) {
        for (size_t i = 0; i < df.size(); ++i) {
            df[i][index] = column[i];
        }
    }

    float calculate_mean(const std::vector<float>& column) {
        float sum = 0.0f;
        int count = 0;
        for(int i=0;i<column.size();i++){
            if(column[i] != -1){
                sum += column[i];
                count ++;
            }
        }
        return sum / count;
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
        int totalSamples = y.size();
        int count0 = 0, count1 = 0;
        
        // Count the number of samples for each class
        for (const auto& label : y) {
            int cls = static_cast<int>(label[0]);  //y contains 0 or 1
            if (cls == 0) count0++;
            else count1++;
        }
        
        // Compute prior probabilities
        priorProb0 = static_cast<float>(count0) / totalSamples;        //計算每種class的機率
        priorProb1 = static_cast<float>(count1) / totalSamples;
        
        int numFeatures = X[0].size();
        
        // Initialize featureProbabilities
        featureProbabilities0.resize(numFeatures);
        featureProbabilities1.resize(numFeatures);
        
        // Calculate conditional probabilities
        for (int i = 0; i < X.size(); ++i) {
            int cls = static_cast<int>(y[i][0]);
            
            for (int j = 0; j < numFeatures; ++j) {
                if (cls == 0) {
                    featureProbabilities0[j][X[i][j]]++;//第 j feature 出現 X[i][j] 的次數+1
                } else {
                    featureProbabilities1[j][X[i][j]]++;//第 j feature 出現 X[i][j] 的次數+1
                }
            }
        }
        // Normalize probabilities
        normalizeProbabilities(featureProbabilities0, count0);
        normalizeProbabilities(featureProbabilities1, count1);
        
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for Naive Bayes classifier
        vector<vector<float>> predictions;

        int count=0;//
        for (const auto& instance : X) {
            float posteriorProb0 = log(priorProb0) + computeLogConditionalProb(instance, featureProbabilities0);
            float posteriorProb1 = log(priorProb1) + computeLogConditionalProb(instance, featureProbabilities1);

            int predictedClass = (posteriorProb1 > posteriorProb0) ? 1 : 0;
            predictions.push_back({static_cast<float>(predictedClass)});
        }

        return predictions;
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for Naive Bayes classifier
        vector<unordered_map<int, float>> probabilities;

        for (const auto& instance : X) {
            float posteriorProb0 = log(priorProb0) + computeLogConditionalProb(instance, featureProbabilities0);
            float posteriorProb1 = log(priorProb1) + computeLogConditionalProb(instance, featureProbabilities1);
            
            float prob0 = exp(posteriorProb0);
            float prob1 = exp(posteriorProb1);

            probabilities.push_back({{0, prob0}, {1, prob1}});
        }

        return probabilities;
    }

private: 
    // Implement private function or variable if you needed
    float priorProb0;
    float priorProb1;
    vector<unordered_map<float, float>> featureProbabilities0;
    vector<unordered_map<float, float>> featureProbabilities1;

    void normalizeProbabilities(vector<unordered_map<float, float>>& probabilities, int count) {
        int i =0;
        for (auto& probMap : probabilities) {
            for (auto& [value, countValue] : probMap) { //value是那個feature的一個值，countValue是value出現過幾次
                if(i>16){
                    countValue = countValue/count;
                }else{
                    countValue = (countValue + 3) / (count + probMap.size() * 3);  // Using a small epsilon
                }
            }
            i++;
        }
    }

    float computeLogConditionalProb(const vector<float>& instance, const vector<unordered_map<float, float>>& featureProbabilities) {
        float logProb = 0.0f;

        for (int j = 0; j < instance.size(); ++j) {
            auto it = featureProbabilities[j].find(instance[j]);
            
            if (it != featureProbabilities[j].end()) {
                logProb += log(it->second);
            } else {
                // Handle cases where the value is not present in training data
                logProb += log(1e-3f);  // Using a small epsilon
                
            }
        }
        return logProb;
    }

};


class KNearestNeighbors: public Classifier {
public:
    KNearestNeighbors(int k = 3): k(k) {} // constructor

    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {
        this->X_train = X;
        this->y_train = y;
    }
    vector<vector<float>> predict(vector<vector<float>> &X) override {
        // Implement the prediction logic for KNN
        std::vector<std::vector<float>> predictions;
        for (const auto& sample : X) {
            auto prediction = _predict_single_sample(sample);
            predictions.push_back({prediction});
        }
        return predictions;
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {
        // Implement probability estimation for KNN
        std::vector<std::unordered_map<int, float>> probabilities;
        for (const auto& sample : X) {
            std::vector<std::pair<float, float>> distances_labels;  // pair of (distance, label)
            // calculate distances between the test sample and all training datas
            for (size_t i = 0; i < X_train.size(); ++i) {
                float dist = _euclidean_distance(sample, X_train[i]);
                distances_labels.push_back({dist, y_train[i][0]});
            }
            // sort based on distances and consider only the closest k datas
            std::sort(distances_labels.begin(), distances_labels.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            // take a vote among the k nearest neighbors
            std::unordered_map<float, int> label_count;
            for (int i = 0; i < k; ++i) {
                label_count[distances_labels[i].second]++;
            }
            // calculate probabilities
            std::unordered_map<int, float> class_probabilities;
            for (const auto& [label, count] : label_count) {
                class_probabilities[static_cast<int>(label)] = static_cast<float>(count) / k;  // convert count to probability
            }
            probabilities.push_back(class_probabilities);
        }

        return probabilities;
    }

private: 
    // Implement private function or variable if you needed
    int k;
    std::vector<std::vector<float>> X_train;
    std::vector<std::vector<float>> y_train;
    float _euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float distance = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            distance += std::pow(a[i] - b[i],2);
        }
        return std::sqrt(distance);
    }
    float _predict_single_sample(const std::vector<float>& sample) {
        std::vector<std::pair<float, float>> distances_labels;  // Pair of (distance, label)
        
        // calculate distances between the test sample and all training datas
        for (size_t i = 0; i < X_train.size(); ++i) {
            float dist = _euclidean_distance(sample, X_train[i]);
            distances_labels.push_back({dist, y_train[i][0]});
        }
        // sort based on distances and consider only the closest k datas
        std::sort(distances_labels.begin(), distances_labels.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        // take a vote among the k nearest neighbors
        std::unordered_map<float, int> label_count;
        for (int i = 0; i < k; ++i) {
            label_count[distances_labels[i].second]++;
        }
        // Find the majority class
        float max_count = 0;
        float predicted_label = -1;
        for (const auto& [label, count] : label_count) {
            if (count > max_count) {
                max_count = count;
                predicted_label = label;
            }
        }
        return predicted_label;
        
    }


};
class MultilayerPerceptron : public Classifier {
public:
    MultilayerPerceptron(int input_size = 64, int hidden_size = 64, int output_size = 64)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), epochs(15), learning_rate((20*0.05/(epochs+20))) {
        initialize_weights();
    }

    void fit(vector<vector<float>>& X, vector<vector<float>>& y) override {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            //learning_rate = (20*learning_rate)/(20+epoch);
            for(size_t i = 0; i<X.size(); i++){   
                vector<vector<float>> input = {X[i]};
                vector<vector<float>> target = {y[i]};
                _forward_propagation(input);
                _backward_propagation(target);
            }
        }
    }

    vector<vector<float>> predict(vector<vector<float>>& X) override {
        vector<vector<float>> predictions;
        for (size_t i = 0; i < X.size(); i++) {
            vector<vector<float>> input = {X[i]};
            _forward_propagation(input);
            vector<float> sample_predictions;
            for (size_t j = 0; j < output_layer.size(); ++j) {
                sample_predictions.push_back((output_layer[j][0] > 0.5) ? 1.0 : 0.0);
            }
            predictions.push_back(sample_predictions);
        }
        return predictions;
    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>>& X) {
        vector<unordered_map<int, float>> probabilities;
        for (size_t i = 0; i < X.size(); i++) {
            vector<vector<float>> input = {X[i]};
            _forward_propagation(input);
            
            unordered_map<int, float> prob_map;
            
            // a binary classification task
            float prob_positive_class = output_layer[0][0];
            float prob_negative_class = 1.0 - prob_positive_class;
            
            prob_map[1] = prob_positive_class; // mapping class 1 to its probability
            prob_map[0] = prob_negative_class; // mapping class 0 to its probability
            
            probabilities.push_back(prob_map); // store the probabilities for this sample
        }
        return probabilities;
    }

private:
    int input_size;
    int hidden_size;
    int output_size;
    int epochs;
    float learning_rate;

    vector<vector<float>> input_layer;
    vector<vector<float>> hidden_layer;
    vector<vector<float>> output_layer;

    vector<vector<float>> weights_input_hidden;
    vector<vector<float>> weights_hidden_output;

    void initialize_weights() {
        // Initialize weights with random values
        weights_input_hidden.resize(input_size, vector<float>(hidden_size));
        weights_hidden_output.resize(hidden_size, vector<float>(output_size));

        std::random_device rd;
        std::default_random_engine generator(rd());
        std::normal_distribution<float> distribution(0.0, 1.0);
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights_input_hidden[i][j] = distribution(generator);
            }
        }

        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights_hidden_output[i][j] = distribution(generator);
            }
        }
    }

    void _forward_propagation(vector<vector<float>>& X) {
        //only a single sample is passed in X.
        input_layer = X;

        // hidden Layer
        hidden_layer.resize(1, vector<float>(hidden_size)); // resizing for one sample
        for (int j = 0; j < hidden_size; ++j) {
            float sum = 0.0;
            for (size_t k = 0; k < input_size; ++k) {
                sum += X[0][k] * weights_input_hidden[k][j];
            }
            hidden_layer[0][j] = sigmoid(sum);
        }
        // output Layer
        output_layer.resize(1, vector<float>(output_size)); // resizing for one sample
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0;
            for (int k = 0; k < hidden_size; ++k) {
                sum += hidden_layer[0][k] * weights_hidden_output[k][j];
            }
            output_layer[0][j] = sigmoid(sum); // as output_layer[j];
        }
        
    }

    void _backward_propagation(vector<vector<float>>& target) {
        // Assuming only a single target value for the output layer
        float output_error = (target[0][0] - output_layer[0][0]) * output_layer[0][0] * (1 - output_layer[0][0]);

        // Error for the hidden layer
        vector<float> hidden_error(hidden_size, 0.0);
        for (int j = 0; j < hidden_size; ++j) {
            float error = output_error * weights_hidden_output[j][0];
            hidden_error[j] = error * hidden_layer[0][j] * (1 - hidden_layer[0][j]); 
        }

        // Update weights between hidden and output layers
        for (int i = 0; i < hidden_size; ++i) {
            float gradient = hidden_layer[0][i] * output_error;
            weights_hidden_output[i][0] += learning_rate * gradient;
        }

        // Update weights between input and hidden layers
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                float gradient = input_layer[0][i] * hidden_error[j];
                weights_input_hidden[i][j] += learning_rate * gradient;
            }
        }

    }

    float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }
};


unordered_map<string, float> evaluate_model(Classifier*, vector<vector<float>>&, vector<vector<float>>&, float);

int main() {
    string train_pth = "trainWithLabel.csv";
    string test_pth = "testWithoutLabel.csv";
    vector<vector<float>> train_df = read_csv_file(train_pth);
    //cout<<"ooooo"<<endl;
    vector<vector<float>> test_df = read_csv_file(test_pth);
    //cout<<train_df.size()<<" "<<train_df[0].size()<<" "<<test_df.size()<<" "<<test_df[0].size()<<endl; 
    /*create dictionary of models for iterating*/
    unordered_map<float, Classifier*> models;
    models[1.0f] = new NaiveBayesClassifier(); // 1.0 represents Naive Bayes
    models[2.0f] = new KNearestNeighbors(); // 2.0 represents KNN 
    models[3.0f] = new MultilayerPerceptron(); // 3.0 represents MLP 
    
    /*preprocessing*/
    Preprocessor test_preprocessor(test_df);
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
            for(int i = 0; i<17; i++){
                max0[i]=0;
                min0[i]=0;
                mean0[i]=0;
            }
            for(int i=0;i<60;i++){
                most0[i]=0;
            }
            auto &train_indices = folds[fold].first;
            auto &val_indices = folds[fold].second;
            // get X_fold, y_fold
            vector<vector<float>> X_train_fold, y_train_fold, X_val_fold, y_val_fold;
            
            for (auto &idx: train_indices) {
                X_train_fold.push_back(X_train[idx]);
                y_train_fold.push_back(y_train[idx]); 
            }

            Preprocessor X_preprocessor(X_train_fold);
            X_train_fold = X_preprocessor.preprocess();

            for (auto &idx: val_indices) {
                X_val_fold.push_back(X_train[idx]);
                y_val_fold.push_back(y_train[idx]);
            }
            validation = true;
            Preprocessor X_val_preprocessor(X_val_fold);
            X_val_fold = X_val_preprocessor.preprocess();
            validation = false;

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
