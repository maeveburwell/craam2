TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += ../../
INCLUDEPATH += /home/marek/R/x86_64-pc-linux-gnu-library/3.6/Rcpp/include
INCLUDEPATH += /usr/include/R
INCLUDEPATH += ../../include
INCLUDEPATH += /home/marek/R/x86_64-pc-linux-gnu-library/3.6/RcppProgress/include

SOURCES += \
    utils.hpp \
    robust_algorithms.cpp \
    simulation.cpp
