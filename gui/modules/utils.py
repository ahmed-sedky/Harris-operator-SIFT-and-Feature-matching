from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import time

import algorithms.harris as Harris
import algorithms.sift as SIFT
import algorithms.match as Match
import modules.message as Message
import modules.image as Image

output_image_path = "cached/output.png"


def browse_files(self, input_image):
    file_name = QFileDialog.getOpenFileName(
        self, "Open file", "./test", "*.jpg;;" " *.png;;" "*.jpeg;;"
    )
    file_path = file_name[0]

    extensionsToCheck = (".jpg", ".png", ".jpeg", ".jfif")
    if file_name[0].endswith(extensionsToCheck):
        start(self, file_path, input_image)
    elif file_name[0] != "":
        Message.error(self, "Invalid format.")
        return
    else:
        return


def start(self, file_path, input_image):
    global original_image_matrix, template_image_matrix

    self.output_image.clear()
    self.template_image.clear()
    plot_image(self, file_path, input_image)
    enable_actions(self)

    try:
        if feature:
            toggle_template_widget(self, True)
    except:
        toggle_template_widget(self, False)
        self.features_combobox.setCurrentIndex(-1)

    if input_image == "original":
        original_image_matrix = Image.read(file_path)

    if input_image == "template":
        template_image_matrix = Image.read(file_path)
        start = time.time()
        if feature == "SSD":
            output_image = Match.SSD(original_image_matrix, template_image_matrix)
        if feature == "NCC":
            output_image = Match.NCC(original_image_matrix, template_image_matrix)
        end = time.time()
        Image.write(output_image_path, output_image)
        plot_image(self, output_image_path, "output")
        Message.info(self, f"Time taken equals {round(end - start, 2)} seconds")


def plot_image(self, image_path, image_type):
    if image_type == "original":
        self.original_image.setPhoto(QPixmap(image_path))
    if image_type == "output":
        self.output_image.setPhoto(QPixmap(image_path))
    if image_type == "template":
        self.template_image.setPhoto(QPixmap(image_path))


def enable_actions(self):
    self.features_combobox.setEnabled(True)


def toggle_template_widget(self, status):
    self.template_widget.setVisible(status)


def choose_feature(self, text):
    global feature

    feature = text
    start = time.time()
    if text == "Harris Response":
        toggle_template_widget(self, False)
        output_image = Harris.harris_response(original_image_matrix)
    elif text == "Corners":
        toggle_template_widget(self, False)
        output_image = Harris.corners(original_image_matrix)
    elif text == "SIFT":
        toggle_template_widget(self, False)
        output_image = SIFT.image(original_image_matrix)
    elif text == "SSD":
        self.output_image.clear()
        try:
            output_image = Match.SSD(original_image_matrix, template_image_matrix)
        except:
            toggle_template_widget(self, True)
            return
    elif text == "NCC":
        self.output_image.clear()
        try:
            output_image = Match.SSD(original_image_matrix, template_image_matrix)
        except:
            toggle_template_widget(self, True)
            return
    end = time.time()
    Image.write(output_image_path, output_image)
    plot_image(self, output_image_path, "output")
    Message.info(self, f"Time taken equals {round(end - start, 2)} seconds")
