#include "controlocr.h"
#include <QString>
#include "neuralnetwork.h"
#include <vector>
#include <iostream>
#include <sstream>
#include "opencv2\opencv.hpp"
#include <QMessageBox>

using namespace std;
using namespace cv;

ControlOCR::ControlOCR()
{
    Neuron::eta = 0.25;   //learning rate
    Neuron::alpha = 0.5; //momentum
    wL = 8; //lebar gambar untuk input training
    hL = 8; //tinggi gambar untuk jadi input training
    minW = 3; //mininal lebar pixel untuk object karakter dari gambar training
    minH = 3; //mininal tinggi pixel untuk object karakter dari gambar training
    maxInterasi = 100000;
    targetError = 0.0001;
    prosesPersen = 0;
    siap = false;
}


string ControlOCR::rekognisi()
{
    stringstream ss;
    string s;
    char c;
    string hasil = "";
    cariBlokObjectKarakter(I, _I, minW, minH);
    for(unsigned int i = 0; i < karakter.size(); ++i){
        for(unsigned j = 0; j < karakter[i].size(); ++j){
            Mat MI = threshold( scaling(karakter[i][j], wL, hL) , 1, 1);
            vector<double> input_ = ControlOCR::mattovector(MI);
            ocrNet.feedForward(input_ );
            ocrNet.getResults(resultVals);
            //tampilVectorVals("result:", resultVals);
            c = targetkarakter[solusi(resultVals)];

            ss.clear();
            ss << c;
            ss >> s;
            hasil += s;

        }
        hasil += "\n";
    }

    return hasil;
}

void ControlOCR::loadHasilTraining()
{
    /***************** Mengambil hasil training ***********************/
    ifstream infile;
    infile.open (namaFilePenyimpanan.c_str(), ios::in);
    if(infile.fail()){
        //cout << "Error : Gagal membuka file..!!" << endl;
        return;
    }

    string line;
    getline(infile, line);
    stringstream ss(line);
    string label;
    ss>> label;


    if (label.compare("topology:") == 0) {
        unsigned value;
        topology.clear();

        while(ss >> value){
            topology.push_back(value);
        }
    }

    getline(infile, line);

    ss.clear();
    ss.str(line);
    ss >> label;


    if (label.compare("wInput:") == 0) {
        unsigned value;
        ss >> value;

        wL = value;
    }
    getline(infile, line);

    ss.clear();
    ss.str(line);
    ss>> label;
    if (label.compare("hInput:") == 0) {
        unsigned value;
        ss >> value;

        hL = value;
    }
    getline(infile, line);
    ss.clear();
    ss.str(line);
    ss>> label;
    while(label.compare("input:") == 0 || label.compare("target:") == 0){
        double value;
        if(label.compare("input:") == 0){
            vector<double> vec;
            while (ss >> value)
            {
                vec.push_back(value);
            }
            input.push_back(vec);
        }else if(label.compare("target:") == 0){
            vector<double> vec;
            while (ss >> value)
            {
                vec.push_back(value);
            }
            target.push_back(vec);
        }

        getline(infile, line);
        ss.clear();
        ss.str(line);
        ss>> label;
    }

    if (label.compare("targetKarakter:") == 0) {
        int value;
        targetkarakter.clear();

        while(ss >> value){
            targetkarakter.push_back((char)value);
        }
    }

    ocrNet.setTopology(topology);

    ocrNet.load(infile);
    infile.close();
}

void ControlOCR::saveHasilTraining(){
    ofstream outfile;
    outfile.open (namaFilePenyimpanan.c_str(), ios::out);

    outfile<<"topology: ";
    for(unsigned int i = 0; i < topology.size(); ++i){
        outfile<<topology[i]<<" ";
    }
    outfile<<endl;

    outfile<<"wInput: "<<wL<<endl;
    outfile<<"hInput: "<<hL<<endl;

    for(unsigned n = 0; n < input.size(); ++n){
        outfile<<"input: ";
        for (unsigned j = 0; j < input[n].size(); j++)
        {
            outfile<< (input[n][j])<<" ";
        }
        outfile<<endl;

        outfile<<"target: ";
        for (unsigned j = 0; j < target[n].size(); j++)
        {
            outfile<< (target[n][j])<<" ";
        }
        outfile<<endl;
    }

    outfile<<"targetKarakter: ";
    for(unsigned int i = 0; i < karakter.size(); ++i){
        outfile<<(int)targetkarakter[i]<<" ";
    }
    outfile<<endl;

    ocrNet.simpan(outfile);

    outfile.close();
}

void ControlOCR::trainingProses()
{
    iterasi = 0;
    do
    {
        for(unsigned n = 0; n < input.size(); ++n){

            ocrNet.feedForward(input[n]);

            ocrNet.backProp(target[n]);
        }
        iterasi++;
        prosesPersen = targetError / ocrNet.getError() * 100;

    }while (ocrNet.getError() > targetError && iterasi < maxInterasi);
}

void ControlOCR::loadImage()
{
    I = imread(namaFileGambar, CV_LOAD_IMAGE_COLOR);
    if(I.empty()){
        //cout << "Error : Gagal membuka gambar..!!" << endl;
        return;
    }

    cariBlokObjectKarakter(I, _I, minW, minH);
}

void ControlOCR::tampilVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int ControlOCR::solusi(vector<double> &v)
{
    unsigned is = 0;
    for (unsigned i = 0; i < v.size(); ++i) {
        if(v[i] >= v[is])
            is = i;
    }

    return is;
}

Mat ControlOCR::rgb2gray(Mat& src)
{
    CV_Assert(src.depth() != sizeof(uchar)); //harus 8 bit

    Mat mc1(src.rows, src.cols, CV_8UC1); //buat matrik 1 chanel
    uchar data;

    if(src.channels() == 3){
        Mat_<Vec3b> _I = src;

        for( int i = 0; i < src.rows; ++i)
            for( int j = 0; j < src.cols; ++j )
            {
                data = (uchar)(_I(i,j)[0] * 0.0722 + _I(i,j)[1] * 0.7152 + _I(i,j)[2] * 0.2126);

                mc1.at<uchar>(i,j) = data;

            }

        src = _I;
        return mc1;

    }else{

        return src;
    }

}


Mat ControlOCR::threshold( Mat src, double thresh, uchar maxval )
{
    for( int i = 0; i < src.rows; ++i)
        for( int j = 0; j < src.cols; ++j )
            src.at<uchar>(i,j) = ((src.at<uchar>(i,j) > thresh)? maxval : 0 );

    return src;
}

vector<double> ControlOCR::mattovector(Mat& src)
{
    vector<double> vec;
    for( int i = 0; i < src.rows; ++i){
        for( int j = 0; j < src.cols; ++j ){
            vec.push_back((double)src.at<uchar>(i,j));
            //cout<<vec.back()<<" ";
        }
        //cout<<endl;
    }
    return vec;
}

Mat ControlOCR::scaling(Mat src, int row, int col)
{
    Mat m =  Mat::zeros(row, col, CV_8UC1);

    for(int i=0; i < m.rows; ++i){
        for(int j=0; j < m.cols; ++j){
            m.at<uchar>(i,j) = src.at<uchar>((i * src.rows / m.rows), (j * src.cols / m.cols));
        }
    }

    return m;
}

void ControlOCR::GarisHorizontal(Mat &src, int y, int xAwal, int xAkhir)
{
    for( int x = xAwal; x < xAkhir; ++x ){
        ((Mat_<Vec3b>)src)(y, x)[0] = 255;
        ((Mat_<Vec3b>)src)(y, x)[1] = 0;
        ((Mat_<Vec3b>)src)(y, x)[2] = 0;
    }
}

void ControlOCR::GarisVertikal(Mat &src, int x, int yAwal, int yAkhir)
{
    for( int y = yAwal; y < yAkhir; ++y ){
        ((Mat_<Vec3b>)src)(y, x)[0] = 255;
        ((Mat_<Vec3b>)src)(y, x)[1] = 0;
        ((Mat_<Vec3b>)src)(y, x)[2] = 0;
    }
}

//[baris][kolom]
void ControlOCR::cariBlokObjectKarakter(Mat gambar, Mat &gambarBlok, int minW, int minH)
{
    vector<Mat> bkarakter;
    Mat tkarakter;
    Mat T, G;

    gambar.copyTo(gambarBlok);
    G = rgb2gray(gambarBlok);

    T = threshold(G, 200, 255);

    bool ketemuAtas = false,
         ketemuBawah = false,
         ketemuKiri = false,
         ketemuKanan = false,
         ketemuKarakterA = false,
         ketemuKarakterB = false;

    int yBAtas, xBKiri, yKAtas, yKBawah;


    for( int i = 0; i < T.rows; ++i){

        ketemuBawah = true;
        for( int j = 0; j < T.cols; ++j ){
            if(!ketemuAtas && T.at<uchar>(i, j) != 255){
                ketemuAtas = true;
                yBAtas = i;

            }else if( ketemuAtas && T.at<uchar>(i, j) != 255){
                ketemuBawah = false;
            }
        }

        if(ketemuAtas && ketemuBawah){
            if(i - minH >= yBAtas){
                GarisHorizontal(gambarBlok, yBAtas, 0, T.cols);
                GarisHorizontal(gambarBlok, i, 0, T.cols);
                ketemuKiri = false;

                for( int j = 0; j < T.cols; ++j ){
                    ketemuKanan = true;
                    for(int ii = yBAtas; ii <= i; ++ii){
                        if( !ketemuKiri && T.at<uchar>(ii, j) != 255){
                            ketemuKiri = true;
                            xBKiri = j;
                        }else if( ketemuKiri && T.at<uchar>(ii, j) != 255 ){
                            ketemuKanan = false;
                        }
                    }

                    if( ketemuKiri && ketemuKanan ){
                        if(j - minW >= xBKiri){
                            //GarisVertikal(I, xBKiri, yBAtas, i);
                            //GarisVertikal(I, j, yBAtas, i);

                            ketemuKarakterA = false;
                            ketemuKarakterB = false;

                            int ii = yBAtas;
                            int jj = xBKiri;

                            while( !ketemuKarakterA && ii <= i){
                                jj = xBKiri;
                                while( !ketemuKarakterA && jj <= j){
                                    if(T.at<uchar>(ii, jj) != 255){
                                        ketemuKarakterA = true;
                                        yKAtas = ii;
                                    }else{
                                        ++jj;
                                    }
                                }
                                ++ii;
                            }

                            ii = i;
                            jj = xBKiri;
                            while( !ketemuKarakterB && ii >= yBAtas){
                                jj = xBKiri;
                                while( !ketemuKarakterB && jj <= j){
                                    if(T.at<uchar>(ii, jj) != 255){
                                        ketemuKarakterB = true;
                                        yKBawah = ii + 1;
                                    }else{
                                        ++jj;
                                    }
                                }
                                --ii;
                            }

                            if(ketemuKarakterA && ketemuKarakterB && (yKBawah-yKAtas) >= minH){
                                GarisHorizontal(gambarBlok, yKAtas, xBKiri, j);
                                GarisHorizontal(gambarBlok, yKBawah, xBKiri, j);
                                GarisVertikal(gambarBlok, xBKiri, yKAtas, yKBawah);
                                GarisVertikal(gambarBlok, j, yKAtas, yKBawah);

                                //simpat gambar karakter ke Mat baru
                                int row = yKBawah - yKAtas,
                                    col = j - xBKiri;
                                tkarakter = Mat::zeros(row, col, CV_8UC1);
                                for(int ci = 0; ci < tkarakter.rows; ++ci){
                                    for(int cj = 0; cj < tkarakter.cols; ++cj){
                                        tkarakter.at<uchar>(ci,cj) = T.at<uchar>(ci + yKAtas, cj + xBKiri );
                                    }
                                }
                                bkarakter.push_back(tkarakter);

                            }


                        }
                        ketemuKiri = false;


                    }

                }

                karakter.push_back(bkarakter);
                bkarakter.clear();
            }

            ketemuAtas = false;
        }
    }
}
