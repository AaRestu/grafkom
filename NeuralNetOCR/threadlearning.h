#ifndef THREADLEARNING_H
#define THREADLEARNING_H

#include <QThread>
#include <QtCore>
#include "controlocr.h"

class ThreadLearning : public QThread
{
    Q_OBJECT
public:
    explicit ThreadLearning(QObject *parent = 0);
    void run();
    bool Stop;
    ControlOCR *cOCR;
signals:
    void LearningProses();
public slots:

};

#endif // THREADLEARNING_H
