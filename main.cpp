#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

class Perceptron {
private:
    vector<double> weights;
    double bias;
    double learning_rate;

public:
    Perceptron(int input_size, double lr = 0.1) {
        learning_rate = lr;
        bias = 0.0;
        // Initialize random weights between -1 and 1
        for (int i = 0; i < input_size; ++i)
            weights.push_back((double(rand()) / RAND_MAX) * 2 - 1);
    }

    int predict(const vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i)
            sum += weights[i] * inputs[i];
        return (sum >= 0) ? 1 : 0;
    }

    void train(const vector<vector<double>>& training_data,
               const vector<int>& labels,
               int epochs = 20) {
        for (int e = 0; e < epochs; ++e) {
            int total_errors = 0;
            for (size_t i = 0; i < training_data.size(); ++i) {
                int prediction = predict(training_data[i]);
                int error = labels[i] - prediction;

                // Update weights and bias
                for (size_t j = 0; j < weights.size(); ++j)
                    weights[j] += learning_rate * error * training_data[i][j];
                bias += learning_rate * error;

                total_errors += abs(error);
            }
            cout << "Epoch " << e + 1 << " | Errors: " << total_errors << endl;
        }
    }

    void showWeights() {
        cout << "\nFinal weights: ";
        for (double w : weights)
            cout << w << " ";
        cout << "\nFinal bias: " << bias << endl;
    }
};

int main() {
    srand(time(0));

    // Training data: (x, y) + label (0 = left side, 1 = right side)
    vector<vector<double>> data = {
        {-2, 1}, {-1, -1}, {1, 2}, {2, -1}, {-1.5, 0.5}, {1.5, -0.5}
    };
    vector<int> labels = {0, 0, 1, 1, 0, 1};

    Perceptron p(2, 0.1);
    p.train(data, labels, 15);
    p.showWeights();

    cout << "\n--- Testing ---" << endl;
    vector<vector<double>> tests = {{-2, -1}, {2, 2}, {0.5, -0.5}};
    for (auto& t : tests)
        cout << "(" << t[0] << ", " << t[1] << ") => "
             << (p.predict(t) ? "Right" : "Left") << endl;
}
