import sys
import cv2
import os
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QDialog, QMessageBox)
from ultralytics import YOLO


import sounddevice as sd
import numpy as np
import whisper as openai_whisper
from scipy.io.wavfile import write

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    update_gestures = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.model = YOLO("training_results/exp_1/weights/last.pt")
        self.detected_gestures = []
        self.frame_count = 0
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                current_gestures = []
                
                if self.frame_count % 10 == 0:
                    results = self.model(frame)
                    if results and len(results) > 0:
                        for result in results:
                            for box in result.boxes:
                                if box.conf[0] > 0.5:
                                    label = self.model.names[int(box.cls[0])]
                                    current_gestures.append(label)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, label, (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                        for gesture in current_gestures:
                            if gesture not in self.detected_gestures:
                                self.detected_gestures.append(gesture)
                        self.update_gestures.emit(self.detected_gestures.copy())

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap.emit(convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio))

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class ReplyVideoPlayer(QDialog):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Reply Video")
        self.setGeometry(100, 100, 640, 480)
        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 10 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        else:
            self.timer.stop()
            self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Communication App")
        self.setGeometry(100, 100, 800, 600)
        
        # UI Components
        self.video_label = QLabel()
        self.gesture_label = QLabel("Detected Gestures: None")
        self.input_field = QLineEdit()
        self.submit_btn = QPushButton("Submit Text")
        self.audio_btn = QPushButton("Record Audio")
        
        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.input_field)
        control_layout.addWidget(self.submit_btn)
        control_layout.addWidget(self.audio_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.gesture_label)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video Thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap.connect(self.update_image)
        self.video_thread.update_gestures.connect(self.update_gesture_list)
        self.video_thread.start()

        # Connections
        self.submit_btn.clicked.connect(self.process_text_input)
        self.audio_btn.clicked.connect(self.process_audio_input)

        self.gesture_vocab =['hello','namaste','no','please','sorry','thanks','yes'] # Add all your gestures
        self.audio_model = openai_whisper.load_model("small") 

        # App Configuration
        self.IMAGE_FOLDER = "seq_folder"
        self.REPLY_FOLDER = "reply"
        os.makedirs(self.REPLY_FOLDER, exist_ok=True)

    def update_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_gesture_list(self, gestures):
        text = "Detected Gestures: " + ", ".join(gestures) if gestures else "No Gestures Detected"
        self.gesture_label.setText(text)
        self.current_gestures = gestures

    def process_text_input(self):
        if hasattr(self, 'current_gestures') and self.current_gestures:
            reply_text = self.input_field.text().strip().lower()
            if reply_text:
                self.generate_reply_video(reply_text.split())
                self.input_field.clear()

    def process_audio_input(self):
        try:
            # Show recording dialog with clearer instructions
            QMessageBox.information(
                self, 
                "Recording", 
                "Please say gesture commands clearly (e.g., 'hello yes namaste')\n\n"
                "Recording for 3 seconds..."
            )
            
            # Audio recording parameters
            fs = 16000  # Sample rate
            seconds = 3  # Duration
            
            # Record with better error handling
            try:
                recording = sd.rec(
                    int(seconds * fs),
                    samplerate=fs,
                    channels=1,
                    dtype='float32'  # Better quality than default
                )
                sd.wait()
            except sd.PortAudioError as pae:
                raise Exception(f"Microphone error: {str(pae)}")
            
            # Save temporary audio file
            temp_file = os.path.join(self.REPLY_FOLDER, "temp_gesture.wav")
            write(temp_file, fs, recording)
            
            # Transcribe with vocabulary hints
            result = self.audio_model.transcribe(
                temp_file,
                language='en',
                initial_prompt=(
                    "The user is speaking gesture commands. "
                    "Possible gestures: hello, yes, no, namaste, thank you. "
                    "Focus only on these words."
                )
            )
            text = result["text"].lower()
            
            # Clean up
            os.remove(temp_file)
            
            # Filter for known gestures only
            gesture_vocab = ['hello','namaste','no','please','sorry','thanks','yes']
            filtered_gestures = [g for g in gesture_vocab if g in text]
            
            if filtered_gestures:
                # Join with spaces for the input field
                gesture_text = " ".join(filtered_gestures)
                self.input_field.setText(gesture_text)
                self.process_text_input()
            else:
                QMessageBox.warning(
                    self,
                    "No Gestures Detected",
                    f"Speech detected but no valid gestures found.\n\n"
                    f"Raw transcription: {text}\n\n"
                    f"Please try again with these gestures: {', '.join(gesture_vocab)}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Audio Processing Error",
                f"Failed to process audio input:\n\n{str(e)}"
            )
    def generate_reply_video(self, gesture_sequence):
        # Create video from images
        output_path = os.path.join(self.REPLY_FOLDER, "reply.mp4")
        self.create_video_from_images(gesture_sequence, output_path)
        
        # Show video player
        self.video_player = ReplyVideoPlayer(output_path)
        self.video_player.exec_()

    def create_video_from_images(self, gesture_list, output_video):
        """Creates a video from images corresponding to detected gestures."""
        # Define fixed video parameters from reference code
        target_width = 640
        target_height = 480
        frame_rate = 10.0
        frame_repeat = 5  # Each image shown for 0.5 seconds (5 frames at 10 FPS)
        codec = 'MJPG'  # Codec used in reference code
        
        image_files = []
        
        # Validate and collect image paths
        for gesture in gesture_list:
            image_path = os.path.join(self.IMAGE_FOLDER, f"{gesture}.jpg")
            if os.path.exists(image_path):
                image_files.append(image_path)
            else:
                print(f"Warning: Image for '{gesture}' not found!")

        if not image_files:
            print("No valid images found, exiting...")
            return

        # Initialize video writer with fixed dimensions
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_video, fourcc, frame_rate, 
                            (target_width, target_height))

        # Process images with error handling
        for file_path in image_files:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not read {file_path}")
                continue
                
            # Resize to fixed dimensions and repeat frames
            img_resized = cv2.resize(img, (target_width, target_height))
            for _ in range(frame_repeat):
                out.write(img_resized)

        out.release()
        print(f"Video saved as {output_video}")


    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# pip install pyqt5 opencv-python ultralytics 
# pip install openai-whisper
# On macOS using Homebrew:
# brew install ffmpeg