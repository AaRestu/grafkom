#ifndef CONTROLOCR_H
#define CONTROLOCR_H

#include <vector>
#include <iostream>
#include <sstream>
#include <QString>
#include "neuralnetwork.h"
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

class ControlOCR
{
public:
    ControlOCR();
    unsigned
        wL, //lebar gambar untuk input training
        hL, //tinggi gambar untuk jadi input training
        minW, //mininal lebar pixel untuk object karakter dari gambar training
        minH, //mininal tinggi pixel untuk object karakter dari gambar training
        maxInterasi,
        iterasi,
        prosesPersen;

    double
        targetError;
    Mat I, _I;

    //inisialisasi topology neural network
    Net ocrNet;
    vector<unsigned> topology;
    vector< vector<double> > input;
    vector< vector<double> > target;
    vector<double> resultVals;
    vector<char> targetkarakter;
    vector< vector<Mat> > karakter;
    string namaFilePenyimpanan;
    string namaFileGambar;
    bool siap;

    void tampilVectorVals(string label, vector< double > &v);
    void cariBlokObjectKarakter(Mat gambar, Mat &gambarBlok, int minW, int minH);
    void loadImage();
    void saveHasilTraining();
    void loadHasilTraining();
    void trainingProses();
    Mat threshold( Mat src, double thresh, uchar maxval );
    Mat scaling(Mat src, int row, int col);
    vector<double> mattovector(Mat& src);
    string rekognisi();

private:
    int solusi(vector<double> &v);
    Mat rgb2gray(Mat& src);
    void GarisHorizontal(Mat &src, int y, int xAwal, int xAkhir);
    void GarisVertikal(Mat &src, int x, int yAwal, int yAkhir);

};

#endif // CONTROLOCR_H
