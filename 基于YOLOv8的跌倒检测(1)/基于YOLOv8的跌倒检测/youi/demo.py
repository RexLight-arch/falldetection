import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)

window = QWidget()
layout = QVBoxLayout(window)

label = QLabel("0/0")
label.setAlignment(Qt.AlignCenter)
label.setMinimumWidth(80)
label.setAutoFillBackground(True)
label.setStyleSheet("color: white !important; background-color: #121a22 !important;")



layout.addWidget(label)

window.show()
sys.exit(app.exec_())
