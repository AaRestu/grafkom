#include "dialoginputbariskarakter.h"
#include "ui_dialoginputbariskarakter.h"

DialogInputBarisKarakter::DialogInputBarisKarakter(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogInputBarisKarakter)
{
    ui->setupUi(this);

}

void DialogInputBarisKarakter::setBarisKe(int i)
{
    QString str = "Masukan Karakter Di Baris ke " + QString::number(i);
    ui->label->setText(str);
    //
    //ui->lineEdit->setText(str_);
}


DialogInputBarisKarakter::~DialogInputBarisKarakter()
{
    delete ui;
}


void DialogInputBarisKarakter::on_pushButton_clicked()
{
    if(1 == ui->lineEdit->text().length()){
        c = ui->lineEdit->text().toLocal8Bit().data()[0];
        this->close();
    }


}
