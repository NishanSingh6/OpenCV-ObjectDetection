# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from detection import detect_cars
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2


class Ui_MainWindow(object):
    def __init__(self):
        self.tmp = None
        self.detectedImg = None
        self.classifierAddress = 'cascade.xml'

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1321, 816)
        MainWindow.setFixedSize(1321, 816)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(530, 0, 321, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Title.setFont(font)
        self.Title.setTextFormat(QtCore.Qt.AutoText)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.open_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_button.setGeometry(QtCore.QRect(280, 590, 141, 51))
        self.open_button.setObjectName("open_button")
        self.ImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImageLabel.setGeometry(QtCore.QRect(160, 140, 521, 431))
        self.ImageLabel.setBaseSize(QtCore.QSize(0, 0))
        self.ImageLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.ImageLabel.setText("")
        self.ImageLabel.setObjectName("ImageLabel")
        self.ImageLabel.resize(500, 400)
        self.ImageLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.ImageLabel_2.setGeometry(QtCore.QRect(690, 140, 521, 431))
        self.ImageLabel_2.setBaseSize(QtCore.QSize(0, 0))
        self.ImageLabel_2.setFrameShape(QtWidgets.QFrame.Box)
        self.ImageLabel_2.setText("")
        self.ImageLabel_2.setObjectName("ImageLabel_2")
        self.ImageLabel_2.resize(500, 400)
        self.detect_button = QtWidgets.QPushButton(self.centralwidget)
        self.detect_button.setEnabled(False)
        self.detect_button.setGeometry(QtCore.QRect(570, 680, 141, 51))
        self.detect_button.setObjectName("detect_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(820, 600, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.car_count_label = QtWidgets.QLabel(self.centralwidget)
        self.car_count_label.setGeometry(QtCore.QRect(1000, 600, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.car_count_label.setFont(font)
        self.car_count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.car_count_label.setObjectName("car_count_label")
        self.save_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_button.setEnabled(False)
        self.save_button.setGeometry(QtCore.QRect(450, 590, 141, 51))
        self.save_button.setObjectName("save_button")
        self.input2 = QtWidgets.QSpinBox(self.centralwidget)
        self.input2.setGeometry(QtCore.QRect(480, 680, 61, 51))
        self.input2.setProperty("value", 5)
        self.input2.setObjectName("input2")
        self.input1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.input1.setGeometry(QtCore.QRect(280, 680, 71, 51))
        self.input1.setProperty("value", 2.0)
        self.input1.setObjectName("input1")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(160, 680, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(360, 680, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.change_button = QtWidgets.QPushButton(self.centralwidget)
        self.change_button.setGeometry(QtCore.QRect(820, 680, 221, 51))
        self.change_button.setObjectName("change_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.open_button.clicked.connect(self.loadImage)
        self.detect_button.clicked.connect(self.detectImage)
        self.save_button.clicked.connect(self.saveImage)
        self.change_button.clicked.connect(self.changeClassifier)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Load Image

    def loadImage(self):
        try:
            self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
            self.image = cv2.imread(self.filename)
            self.setPhoto(self.image)
            self.detect_button.setEnabled(True)
        except:
            self.detect_button.setEnabled(False)

    def setPhoto(self, image):
        self.tmp = image
        image = cv2.resize(image, (500, 300))
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ImageLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def detectImage(self):
        scale = self.input1.value()
        mN = self.input2.value()

        image, count = detect_cars(self.tmp, scale, mN, self.classifierAddress)
        self.detectedImg = image
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ImageLabel_2.setPixmap(QtGui.QPixmap.fromImage(image))
        self.car_count_label.setText(str(count))
        self.save_button.setEnabled(True)

    def saveImage(self):
        file_name, _ = QFileDialog.getSaveFileName(filter="Image JPEG Files (*.jpg);;PNG Files (*.png)")
        if file_name:
            cv2.imwrite(file_name, self.detectedImg)

    def changeClassifier(self):
        try:
            f = QFileDialog.getOpenFileName(filter="XML (*.xml)")[0]
            self.classifierAddress = f
        except:
            self.classifierAddress = 'cascade.xml'
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Object Detection"))
        self.Title.setText(_translate("MainWindow", "OpenCV - Object Detection"))
        self.open_button.setText(_translate("MainWindow", "Open Image"))
        self.detect_button.setText(_translate("MainWindow", "Detect Cars"))
        self.label.setText(_translate("MainWindow", "Number Of Cars"))
        self.car_count_label.setText(_translate("MainWindow", "0"))
        self.save_button.setText(_translate("MainWindow", "Save Image"))
        self.label_2.setText(_translate("MainWindow", "scaleFactor"))
        self.label_3.setText(_translate("MainWindow", "minNeighbors"))
        self.change_button.setText(_translate("MainWindow", "Change cascade classifier"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
