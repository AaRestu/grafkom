#include "neuralnetwork.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>

double Neuron::eta = 0.25;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeight(Layer &layerSebelumnya)
{
    for (unsigned n = 0; n < layerSebelumnya.size(); ++n) {
        Neuron &neuron = layerSebelumnya[n];
        double deltaWeightDulu = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double deltaWeightBaru =
                eta
                * neuron.getOutputVal()
                * m_gradient
                + alpha
                * deltaWeightDulu;

        neuron.m_outputWeights[m_myIndex].deltaWeight = deltaWeightBaru;
        neuron.m_outputWeights[m_myIndex].weight += deltaWeightBaru;
    }
}

double Neuron::sumDOW(const Layer &layerSelanjutnya) const
{
    double sum = 0.0;

    // Sum [w * gradien]

    for (unsigned n = 0; n < layerSelanjutnya.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * layerSelanjutnya[n].m_gradient;
    }

    return sum;
}

void Neuron::hitungHiddenGradien(const Layer &layerSelanjutnya)
{
    double dow = sumDOW(layerSelanjutnya);
    m_gradient = dow * Neuron::transferFunctionTurunan(m_outputVal);
}

void Neuron::hitungOutputGradien(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionTurunan(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    //sigmoid
    return 1 / (1 + exp(-1 * x));
}

double Neuron::transferFunctionTurunan(double x)
{
    //turunan sigmoid
    return (1 / (1 + exp(-1 * x))) * ( 1 - ( 1 / (1 + exp(-1 * x)) ) );
}

void Neuron::feedForward(const Layer &layerSebelumnya)
{
    double sum = 0.0;

    // penjumlahan layer sebelumnya (input termasuk bias)

    for (unsigned n = 0; n < layerSebelumnya.size(); ++n) {
        sum += layerSebelumnya[n].getOutputVal() *
                layerSebelumnya[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = weightAcak();
    }

    m_myIndex = myIndex;
}



void Net::load(ifstream &infile)
{
    string line;
    getline(infile, line);
    stringstream ss(line);
    string label;


    ss>> label;
    if (label.compare("eta:") == 0) {
        double value;
        ss >> value;
        Neuron::eta = value;
    }

    getline(infile, line);
    ss.clear();
    ss.str(line);

    ss>> label;
    if (label.compare("alpha:") == 0) {
        double value;
        ss >> value;
        Neuron::alpha = value;
    }

    getline(infile, line);
    ss.clear();
    ss.str(line);
    ss>> label;

    if (label.compare("net:") == 0) {
        double value;

        for (unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum) {
            Layer &layer = m_layers[layerNum];

            for (unsigned n = 0; n < layer.size() - 1; ++n) {
                ss >> value;
                layer[n].setOutputVal(value);
                ss >> value;
                layer[n].setGradient(value);

                vector< Connection > c = layer[n].getOutputWeights();
                Connection con;
                vector<Connection> outputWeights;
                for (unsigned no = 0; no < c.size(); ++no)
                {
                    ss >> value;
                    con.deltaWeight = value;
                    ss >> value;
                    con.weight = value;
                    outputWeights.push_back(con);
                }
                layer[n].setOutputWeights(outputWeights);
            }
        }
    }

    getline(infile, line);
    ss.clear();
    ss.str(line);
    ss>> label;
    if (label.compare("error:") == 0) {
        double value;
        while (ss >> value) {
            m_error = value;
        }
    }
}

void Net::simpan(ofstream &outfile)
{
    outfile<<"eta: "<<Neuron::eta<<endl;
    outfile<<"alpha: "<<Neuron::alpha<<endl;

    outfile<<"net: ";
    for (unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum) {
        Layer &layer = m_layers[layerNum];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            outfile<<layer[n].getOutputVal()<<" ";
            outfile<<layer[n].getGradient()<<" ";

            vector<Connection> c = layer[n].getOutputWeights();

            for (unsigned no = 0; no < c.size(); ++no)
            {
                outfile<<c[no].deltaWeight<<" ";
                outfile<<c[no].weight<<" ";
            }

        }
    }
    outfile<<endl;

    outfile<<"error: "<<m_error<<endl;
}



void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Perhitungan error net

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;  // Mean Squared Error (MSE)
    m_error = sqrt(m_error); // RMS



    // Perhitungan gradient output layer

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].hitungOutputGradien(targetVals[n]);
    }

    // Perhitungan gradient hidden layer

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &layerSelanjutnya = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].hitungHiddenGradien(layerSelanjutnya);
        }
    }

    // update weight

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &layerSebelumnya = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeight(layerSebelumnya);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Inisialisasi input layer ke setiap neuron di layer pertama
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &layerSebelumnya = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(layerSebelumnya);
        }
    }
}

void Net::setTopology(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Buat neuron dan bias untuk setiap layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            //cout<<"neuron "<<neuronNum<<" ";
        }
        //cout<<endl;



        // isi setiap nilai bias dgn 1 :
        m_layers.back().back().setOutputVal(1.0);
    }
}
