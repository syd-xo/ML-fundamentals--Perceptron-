#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

// A simple implementation of a perceptron — one of the earliest forms of a neural network
class Perceptron {
private:
    vector<double> weights;     // Stores the weights for each input feature
    double bias;                // The bias term
    double learning_rate;       // Controls how much weights are adjusted per iteration

public:
    // Constructor: initializes weights and learning rate
    Perceptron(int input_size, double lr = 0.1) {
        learning_rate = lr;
        bias = 0.0;
        // Initialize random weights between -1 and 1
        for (int i = 0; i < input_size; ++i)
            weights.push_back((double(rand()) / RAND_MAX) * 2 - 1);
    }

    // Predicts the output (0 or 1) given a vector of inputs
    int predict(const vector<double>& inputs) {
        double sum = bias;
        // Compute the weighted sum of inputs
        for (size_t i = 0; i < weights.size(); ++i)
            sum += weights[i] * inputs[i];
        // Apply the step activation function
        return (sum >= 0) ? 1 : 0;
    }

    // Trains the perceptron on the given dataset
    void train(const vector<vector<double>>& training_data,
               const vector<int>& labels,
               int epochs = 20) {
        // Repeat the training process for a number of epochs
        for (int e = 0; e < epochs; ++e) {
            int total_errors = 0;

            // Loop through each training example
            for (size_t i = 0; i < training_data.size(); ++i) {
                int prediction = predict(training_data[i]);
                int error = labels[i] - prediction;

                // Update each weight based on the perceptron learning rule
                for (size_t j = 0; j < weights.size(); ++j)
                    weights[j] += learning_rate * error * training_data[i][j];

                // Update bias
                bias += learning_rate * error;

                // Count total number of errors for this epoch
                total_errors += abs(error);
            }

            // Print the number of errors after each training iteration
            cout << "Epoch " << e + 1 << " | Errors: " << total_errors << endl;
        }
    }

    // Prints the final weights and bias after training
    void showWeights() {
        cout << "\nFinal weights: ";
        for (double w : weights)
            cout << w << " ";
        cout << "\nFinal bias: " << bias << endl;
    }
};

int main() {
    srand(time(0)); // Seed for random weight initialization

    // Training data:
    // Each item is (x, y) followed by its label
    // Label 0 = "left side", Label 1 = "right side"
    vector<vector<double>> data = {
        {-2, 1}, {-1, -1}, {1, 2}, {2, -1}, {-1.5, 0.5}, {1.5, -0.5}
    };
    vector<int> labels = {0, 0, 1, 1, 0, 1};

    // Create a perceptron with 2 inputs and a learning rate of 0.1
    Perceptron p(2, 0.1);

    // Train the perceptron for 15 epochs
    p.train(data, labels, 15);

    // Display the final learned weights and bias
    p.showWeights();

    // Test the perceptron with new unseen points
    cout << "\n--- Testing ---" << endl;
    vector<vector<double>> tests = {
        {-2, -1},
        {2, 2},
        {0.5, -0.5}
    };

    // Predict and display the classification for each test point
    for (auto& t : tests)
        cout << "(" << t[0] << ", " << t[1] << ") => "
             << (p.predict(t) ? "Right" : "Left") << endl;
}
