#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


#include <vector>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace std;


struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
    static double eta;
    static double alpha;
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &layerSebelumnya);
    void hitungOutputGradien(double targetVal);
    void hitungHiddenGradien(const Layer &layerSelanjutnya);
    void updateInputWeight(Layer &layerSebelumnya);
    vector<Connection> getOutputWeights(void) const { return m_outputWeights; }
    void setOutputWeights(vector<Connection> val) { m_outputWeights = val; }
    double getGradient(void) const { return m_gradient; }
    void setGradient(double val) { m_gradient = val; }

private:
    static double transferFunction(double x);
    static double transferFunctionTurunan(double x);
    static double weightAcak(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &layerSelanjutnya) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

// ****************** class Net ******************
class Net
{
public:
    void setTopology(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getError(void) const { return m_error; }
    void simpan(ofstream &outfile);
    void load(ifstream &infile);
private:
    vector<Layer> m_layers;
    double m_error;
};

#endif // NEURALNETWORK_H
