#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include "controlocr.h"
#include "opencv2/opencv.hpp"
#include "threadlearning.h"
#include "dialoginputbariskarakter.h"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cOCR = new ControlOCR();
    ui->spinBoxMinW->setValue(cOCR->minW);
    ui->spinBoxMinH->setValue(cOCR->minH);
    ui->spinBoxWL->setValue(cOCR->wL);
    ui->spinBoxHL->setValue(cOCR->hL );
    ui->spinBoxMaxInterasi->setValue(cOCR->maxInterasi);
    ui->doubleSpinBoxTargetError->setValue(cOCR->targetError);
    ui->doubleSpinBoxEta->setValue(Neuron::eta);
    ui->doubleSpinBoxAlpha->setValue(Neuron::alpha);

    threadLearning = new ThreadLearning(this);
    threadLearning->Stop = true;
    connect(threadLearning, SIGNAL(LearningProses()), this, SLOT(on_ProsesLearning()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_Image_triggered()
{
    QString namafile = QFileDialog::getOpenFileName(this, tr("Open File Image"), "", tr("Gambar (*.jpg)"));
    cOCR->namaFileGambar = namafile.toUtf8().constData();
    cOCR->loadImage();
    if(!cOCR->I.empty() && !cOCR->_I.empty()){
        cv::Mat rgb;
        cv::cvtColor(cOCR->I, rgb, CV_BGR2RGB);

        QImage QimageI((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, QImage::Format_RGB888);
        ui->label_pic->setPixmap(QPixmap::fromImage(QimageI));

        cv::cvtColor(cOCR->_I, rgb, CV_BGR2RGB);
        QImage QimageI_((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, QImage::Format_RGB888);
        ui->label_pic_blok->setPixmap(QPixmap::fromImage(QimageI_));

        ui->groupBox_blok->setEnabled(true);
    }else{
        ui->groupBox_blok->setEnabled(false);
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    cOCR->minW = ui->spinBoxMinW->value();
    cOCR->minH = ui->spinBoxMinH->value();

    cOCR->cariBlokObjectKarakter(cOCR->I, cOCR->_I, cOCR->minW, cOCR->minH);

    cv::Mat rgb;
    cv::cvtColor(cOCR->I, rgb, CV_BGR2RGB);

    QImage QimageI((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, QImage::Format_RGB888);
    ui->label_pic->setPixmap(QPixmap::fromImage(QimageI));

    cv::cvtColor(cOCR->_I, rgb, CV_BGR2RGB);
    QImage QimageI_((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, QImage::Format_RGB888);
    ui->label_pic_blok->setPixmap(QPixmap::fromImage(QimageI_));

}

void MainWindow::on_actionOpen_Hasil_Learning_triggered()
{
    QString namafile = QFileDialog::getOpenFileName(this, tr("Open File Hasil Learning"), "", tr("Files (*.txt)"));
    cOCR->namaFilePenyimpanan = namafile.toUtf8().constData();
    cOCR->loadHasilTraining();
    cOCR->siap = true;

    ui->spinBoxMinW->setValue(cOCR->minW);
    ui->spinBoxMinH->setValue(cOCR->minH);
    ui->spinBoxWL->setValue(cOCR->wL);
    ui->spinBoxHL->setValue(cOCR->hL );
    ui->spinBoxMaxInterasi->setValue(cOCR->maxInterasi);
    ui->doubleSpinBoxTargetError->setValue(cOCR->targetError);
    ui->doubleSpinBoxEta->setValue(Neuron::eta);
    ui->doubleSpinBoxAlpha->setValue(Neuron::alpha);

    ui->pushButtonRekognisi->setEnabled(true);
}

void MainWindow::on_pushButtonRekognisi_clicked()
{
    on_pushButton_2_clicked();
    std::string hasil_= cOCR->rekognisi();
    QString hasil = QString::fromUtf8(hasil_.c_str());
    ui->plainTextEditHasil->setPlainText("");
    ui->plainTextEditHasil->setPlainText(hasil);
}

void MainWindow::on_pushButtonTraining_clicked()
{

    if(threadLearning->Stop){
        if(!cOCR->I.empty() && !cOCR->_I.empty() && !cOCR->siap ){

            char c;
            vector<double> dtarget;
            QString judul = "terdapat " + QString::number(cOCR->karakter.size()) + " baris karakter!";
            for(unsigned int i = 0; i < cOCR->karakter.size(); ++i){
                //inisialisasi karakter perbaris gambar input training
                DialogInputBarisKarakter dibk(this);
                dibk.setWindowTitle(judul);
                dibk.setBarisKe(i+1);
                dibk.exec();


                c = dibk.getC();
                cOCR->targetkarakter.push_back(c);


                //inisialisasi target untuk neural network
                dtarget.clear();
                for(unsigned int io = 0; io < cOCR->karakter.size(); ++io){
                    dtarget.push_back((i == io)? 1.0 : 0.0);
                }


                //inisialisasi input untuk neural network
                for(unsigned int n = 0; n < cOCR->karakter[i].size(); ++n){
                    //scaling kemudian threshold [0,1] lalu dimasukan ke vektor input
                    Mat MI = cOCR->threshold( cOCR->scaling( cOCR->karakter[i][n], cOCR->wL, cOCR->hL) , 1, 1);
                    vector<double> input_ = cOCR->mattovector(MI);
                    cOCR->input.push_back( input_ );

                    cOCR->target.push_back(dtarget);
                }
            }

            cOCR->topology.clear();
            cOCR->topology.push_back(cOCR->wL * cOCR->wL);
            cOCR->topology.push_back(100);
            cOCR->topology.push_back(cOCR->targetkarakter.size());
            cOCR->ocrNet.setTopology(cOCR->topology);
        }
        cOCR->wL = ui->spinBoxWL->value();
        ui->spinBoxWL->setEnabled(false);
        cOCR->hL = ui->spinBoxHL->value();
        ui->spinBoxHL->setEnabled(false);
        cOCR->maxInterasi = ui->spinBoxMaxInterasi->value();
        ui->spinBoxMaxInterasi->setEnabled(false);
        cOCR->targetError = ui->doubleSpinBoxTargetError->value();
        ui->doubleSpinBoxTargetError->setEnabled(false);
        Neuron::eta = ui->doubleSpinBoxEta->value();
        ui->doubleSpinBoxEta->setEnabled(false);
        Neuron::alpha = ui->doubleSpinBoxAlpha->value();
        ui->doubleSpinBoxAlpha->setEnabled(false);

        threadLearning->cOCR = cOCR;
        threadLearning->Stop = false;

        ui->progressBar->setEnabled(true);

        threadLearning->start();
        ui->pushButtonTraining->setText("Stop Training");
    }else{
        ui->spinBoxWL->setEnabled(true);
        ui->spinBoxHL->setEnabled(true);
        ui->spinBoxMaxInterasi->setEnabled(true);
        ui->doubleSpinBoxTargetError->setEnabled(true);
        ui->doubleSpinBoxEta->setEnabled(true);
        ui->doubleSpinBoxAlpha->setEnabled(true);
        ui->progressBar->setEnabled(false);
        ui->progressBar->setValue(0);
        threadLearning->Stop = true;
        ui->pushButtonTraining->setText("Mulai Training");
        if(cOCR->siap)
            ui->pushButtonRekognisi->setEnabled(true);
    }
}

void MainWindow::on_ProsesLearning(){
    ui->progressBar->setValue(this->cOCR->prosesPersen);
    if(this->cOCR->prosesPersen >= 100){
        cOCR->siap = true;
        on_pushButtonTraining_clicked();
    }
}

void MainWindow::on_actionSave_Hasil_Learning_triggered()
{
    QString namafile = QFileDialog::getSaveFileName(this, tr("Open File Hasil Learning"), "", tr("Files (*.txt)"));
    cOCR->namaFilePenyimpanan = namafile.toUtf8().constData();
    cOCR->saveHasilTraining();
}
