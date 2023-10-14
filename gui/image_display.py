
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2


CAPTURE_SIZE = (1280, 720)
DEFAULT_VIEW_SIZE = (640, 360)
VIEW_ASPECT_RATIO = round(DEFAULT_VIEW_SIZE[0] / DEFAULT_VIEW_SIZE[1], 2)


class WorkerThread(QtCore.QThread):
    
    def __init__(self, parent: 'ImageDisplayWidget'):
        super().__init__()
        self._parent = parent
    
    def run(self) -> None:
        while True:
            _, cv2_image = self._parent.capture.read()
            if cv2_image is not None:
                qt5_image = QtGui.QImage(cv2_image.data, *CAPTURE_SIZE, QtGui.QImage.Format.Format_BGR888)
                qt5_image = qt5_image.scaled(*self._parent.view_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self._parent.setPixmap(QtGui.QPixmap.fromImage(qt5_image))


class ImageDisplayWidget(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()

        self.view_size = DEFAULT_VIEW_SIZE
        self._resizing = False

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

        self.worker = WorkerThread(self)
        self.worker.start()

        self.resize(*self.view_size)
    
    def close(self) -> bool:
        self.worker.terminate()
        return super().close()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        return super().resizeEvent(event)

        old_size = event.oldSize()
        if old_size == QtCore.QSize(-1, -1):
            return

        if self._resizing:
            return

        self._resizing = True

        new_size = event.size()
        size_dif = old_size - new_size

        self.view_size = (new_size.width(), new_size.height())

        self.resize(new_size)

        self._resizing = False



if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    win = ImageDisplayWidget()
    win.show()

    app.exec()
