
from image_display import ImageDisplayWidget
from sign_lang_display import SignLangDisplay
from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self._sub_splitter = QtWidgets.QSplitter()
        self._sub_splitter.addWidget(ImageDisplayWidget())
        
        self._main_splitter = QtWidgets.QSplitter()
        self._main_splitter.addWidget(self._sub_splitter)
        self._main_splitter.addWidget(SignLangDisplay())

        self._sub_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self._main_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self._main_splitter.setSizes([4000, 1000])

        self.setCentralWidget(self._main_splitter)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    
    app.exec()
