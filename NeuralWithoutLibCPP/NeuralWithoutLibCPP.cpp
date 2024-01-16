#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

const int inputSize = 2;
const int hiddenSize = 3;
const int outputSize = 1;

double weightInputHidden[inputSize][hiddenSize];
double biasHidden[hiddenSize];

double weightHiddenOutput[hiddenSize][outputSize];
double biasOutput[outputSize];

void initializeWeights()
{
	srand(static_cast<unsigned>(time(NULL)));

	for (int i = 0; i < inputSize; i++)
	{
		for (int j = 0; j < hiddenSize; j++)
		{
			weightInputHidden[i][j] = (rand() % 100) / 100.0;
		}
	}

	for (int i = 0; i < hiddenSize; i++)
	{
		biasHidden[i] = (rand() % 100) / 100.0;
	}

	for (int i = 0; i < hiddenSize; i++)
	{
		for (int j = 0; j < outputSize; j++)
		{
			weightHiddenOutput[i][j] = (rand() % 100) / 100.0;
		}
	}

	for (int i = 0; i < outputSize; i++)
	{
		biasOutput[i] = (rand() % 100) / 100.0;
	}
}

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void forwardPass(double input[inputSize], double hiddenLayer[hiddenSize], double outputLayer[outputSize]) 
{
    for (int i = 0; i < hiddenSize; ++i) 
    {
        hiddenLayer[i] = 0;
        for (int j = 0; j < inputSize; ++j) 
        {
            hiddenLayer[i] += input[j] * weightInputHidden[j][i];
        }
        hiddenLayer[i] += biasHidden[i];
        hiddenLayer[i] = sigmoid(hiddenLayer[i]);
    }

    for (int i = 0; i < outputSize; ++i) 
    {
        outputLayer[i] = 0;
        for (int j = 0; j < hiddenSize; ++j) 
        {
            outputLayer[i] += hiddenLayer[j] * weightHiddenOutput[j][i];
        }
        outputLayer[i] += biasOutput[i];
        outputLayer[i] = sigmoid(outputLayer[i]);
    }
}

void backpropagation(double input[inputSize], double hiddenLayer[hiddenSize], double outputLayer[outputSize], double target) 
{
    double outputError = outputLayer[0] - target;
    double outputDelta = outputError * outputLayer[0] * (1 - outputLayer[0]);

    for (int i = 0; i < hiddenSize; ++i) 
    {
        weightHiddenOutput[i][0] -= outputDelta * hiddenLayer[i];
    }
    biasOutput[0] -= outputDelta;

    double hiddenErrors[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) 
    {
        hiddenErrors[i] = outputDelta * weightHiddenOutput[i][0];
    }

    for (int i = 0; i < hiddenSize; ++i) 
    {
        double hiddenDelta = hiddenErrors[i] * hiddenLayer[i] * (1 - hiddenLayer[i]);
        for (int j = 0; j < inputSize; ++j) 
        {
            weightInputHidden[j][i] -= hiddenDelta * input[j];
        }
        biasHidden[i] -= hiddenDelta;
    }
}

double feedforward(double input[inputSize]) 
{
    double hiddenLayer[hiddenSize];
    double outputLayer[outputSize];

    // Warstwa ukryta
    for (int i = 0; i < hiddenSize; ++i) 
    {
        hiddenLayer[i] = 0;
        for (int j = 0; j < inputSize; ++j) 
        {
            hiddenLayer[i] += input[j] * weightInputHidden[j][i];
        }
        hiddenLayer[i] += biasHidden[i];
        hiddenLayer[i] = sigmoid(hiddenLayer[i]);
    }

    // Warstwa wyjściowa
    for (int i = 0; i < outputSize; ++i) 
    {
        outputLayer[i] = 0;
        for (int j = 0; j < hiddenSize; ++j) 
        {
            outputLayer[i] += hiddenLayer[j] * weightHiddenOutput[j][i];
        }
        outputLayer[i] += biasOutput[i];
        outputLayer[i] = sigmoid(outputLayer[i]);
    }

    return outputLayer[0];
}

// Funkcja trenująca sieć
void train(double input[inputSize], double target) 
{
    double hiddenLayer[hiddenSize];
    double outputLayer[outputSize];

    // Przekazywanie sygnału w przód
    forwardPass(input, hiddenLayer, outputLayer);

    // Algorytm wstecznej propagacji błędu - Aktualizacja wag i biasów
    backpropagation(input, hiddenLayer, outputLayer, target);
}


double calculateError(double predicted, double target)
{
	return 0.5 * pow(predicted - target, 2);
}

int main() {
    initializeWeights();

    double trainingData[][inputSize] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    double targetData[] = { 0, 1, 1, 0 };

    // Trenowanie sieci
    for (int epoch = 0; epoch < 10000; ++epoch) 
    {
        for (int i = 0; i < 4; ++i) 
        {
            train(trainingData[i], targetData[i]);
        }
    }

    // Wyświetlanie wyników po trenowaniu
    cout << "Wyniki po trenowaniu:" << endl;
    for (int i = 0; i < 4; ++i) 
    {
        double predicted = feedforward(trainingData[i]);
        double error = calculateError(predicted, targetData[i]);
        cout << "Input: {" << trainingData[i][0] << ", " << trainingData[i][1]
            << "} Target: " << targetData[i] << " Predicted: " << predicted
            << " Error: " << error << endl;
    }

    // Testowanie na nowych danych
    double testData[][inputSize] = { {0.5, 0.5}, {0.2, 0.8}, {0.8, 0.2}, {0.3, 0.7} };

    cout << "\nWyniki na nowych danych:" << endl;
    for (int i = 0; i < 4; ++i) 
    {
        double predicted = feedforward(testData[i]);
        cout << "Input: {" << testData[i][0] << ", " << testData[i][1]
            << "} Predicted: " << predicted << endl;
    }

    return 0;
}