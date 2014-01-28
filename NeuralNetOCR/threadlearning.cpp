#include "threadlearning.h"
#include <QtCore>

ThreadLearning::ThreadLearning(QObject *parent) :
    QThread(parent)
{
}

void ThreadLearning::run(){
    cOCR->iterasi = 0;
    do
    {
        for(unsigned n = 0; n < cOCR->input.size(); ++n){

            cOCR->ocrNet.feedForward(cOCR->input[n]);

            cOCR->ocrNet.backProp(cOCR->target[n]);
        }
        cOCR->iterasi++;
        cOCR->prosesPersen = cOCR->targetError / cOCR->ocrNet.getError() * 100;
        QMutex mutex;
        mutex.lock();
        if(this->Stop) break;
        mutex.unlock();

        emit LearningProses();
    }while (cOCR->ocrNet.getError() > cOCR->targetError && cOCR->iterasi < cOCR->maxInterasi);
}
