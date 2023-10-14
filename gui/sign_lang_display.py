
from PyQt5 import QtWidgets, QtCore, QtGui


class ImageCard(QtWidgets.QWidget):

    def __init__(self, character: str):
        super().__init__()

        img_fp = f"dataset\\asl_alphabet_test\\asl_alphabet_test\\{character}_test.jpg"
        pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(img_fp))

        image_label = QtWidgets.QLabel()
        char_label = QtWidgets.QLabel(character)
        image_label.setPixmap(pixmap)

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.addWidget(char_label)
        self._layout.addWidget(image_label)
        self.setLayout(self._layout)


class SignLangDisplay(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        
        self.input_line = QtWidgets.QLineEdit()
        self.image_list_widget = QtWidgets.QListWidget()

        self.input_line.returnPressed.connect(self._update_images)

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self.input_line)
        self._layout.addWidget(self.image_list_widget)

        self.setLayout(self._layout)
    
    def _update_images(self) -> None:
        text = self.input_line.text()

        self.image_list_widget.clear()

        for char in text:
            list_item = QtWidgets.QListWidgetItem()
            card = ImageCard(char)
            list_item.setSizeHint(QtCore.QSize(200, 200))

            self.image_list_widget.addItem(list_item)
            self.image_list_widget.setItemWidget(list_item, card)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    
    win = SignLangDisplay()
    win.resize(300, 600)
    win.show()
    
    app.exec()
