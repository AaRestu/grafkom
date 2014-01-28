#-------------------------------------------------
#
# Project created by QtCreator 2014-01-28T15:14:35
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NeuralNetOCR
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    neuralnetwork.cpp \
    controlocr.cpp \
    threadlearning.cpp \
    dialoginputbariskarakter.cpp

HEADERS  += mainwindow.h \
    neuralnetwork.h \
    controlocr.h \
    threadlearning.h \
    dialoginputbariskarakter.h

FORMS    += mainwindow.ui \
    dialoginputbariskarakter.ui

INCLUDEPATH += D://opencv//sources//opencv_bin//install//include

LIBS += D://opencv//sources//opencv_bin//bin//*.dll
