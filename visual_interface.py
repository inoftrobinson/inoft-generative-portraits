import PyQt5
from qt_ui import qt_interface_app_file
import sys, cv2
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

done = False
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_completed = False
        self.ui = qt_interface_app_file.Ui_MainWindow()
        self.ui.setupUi(self)

        self.window_width = self.ui.widget_to_display_realtime_images.frameSize().width()
        self.window_height = self.ui.widget_to_display_realtime_images.frameSize().height()
        self.ui.widget_to_display_realtime_images = OwnImageWidget(self.ui.widget_to_display_realtime_images)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        """
        print("hey")
        vid = cv2.VideoCapture("quel language pour l'intelligence artificielle - rendu 1.mp4")
        ret, frame = vid.read()
        while ret:
            print("hoy")
            # Qimg = PyQt5.QtWidgets.QLabel.
            self.ui.label_to_display_realtime_images.setpixmap(Qimg)
            self.ui.label_to_display_realtime_images.update()
            ret, frame = vid.read()
        """

    def update_frame(self):
        # if not q.empty():
        # frame = q.get()
        vid = cv2.VideoCapture("quel language pour l'intelligence artificielle - rendu 1.mp4")
        ret, frame = vid.read()
        global  done
        while ret and not done:
            done = True
            img_height, img_width, img_colors = frame.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, bpc = frame.shape
            bpl = bpc * width
            image = QtGui.QImage(frame.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ui.widget_to_display_realtime_images.setImage(image)


def except_hook(cls, exception, traceback):
    # Allow to print errors message, otherwise Qt
    # would not print the reasons for a crash
    sys.__excepthook__(cls, exception, traceback)

try:
    from live_on_webcam import *
except:
    from .live_on_webcam import *

def start_app():
    create_matplotlib_window()
    networkSystem = NetworkSystem()
    networkSystem.start_network_loop()

    app = QApplication(sys.argv)
    sys.excepthook = except_hook

    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_app()
