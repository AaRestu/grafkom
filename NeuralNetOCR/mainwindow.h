#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "controlocr.h"
#include "threadlearning.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    ThreadLearning* threadLearning;

private slots:
    void on_actionOpen_Image_triggered();

    void on_pushButton_2_clicked();

    void on_actionOpen_Hasil_Learning_triggered();

    void on_pushButtonRekognisi_clicked();

    void on_pushButtonTraining_clicked();

    void on_ProsesLearning();

    void on_actionSave_Hasil_Learning_triggered();

private:
    Ui::MainWindow *ui;
    ControlOCR* cOCR;
};

#endif // MAINWINDOW_H
