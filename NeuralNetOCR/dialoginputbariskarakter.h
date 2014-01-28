#ifndef DIALOGINPUTBARISKARAKTER_H
#define DIALOGINPUTBARISKARAKTER_H

#include <QDialog>

namespace Ui {
class DialogInputBarisKarakter;
}

class DialogInputBarisKarakter : public QDialog
{
    Q_OBJECT

public:
    explicit DialogInputBarisKarakter(QWidget *parent = 0);
    ~DialogInputBarisKarakter();
    char getC(){return c;}
    void setBarisKe(int i);

private slots:


    void on_pushButton_clicked();

private:
    Ui::DialogInputBarisKarakter *ui;
    char c;
};

#endif // DIALOGINPUTBARISKARAKTER_H
