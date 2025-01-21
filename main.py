import sys
import cv2
import psutil
import time
import threading
import queue
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

# SKU mapping
SKU_MAPPING = {
    "wraps": 1601,
    "salads": 1101,
    "pudding": 2706,
    "yogurt": 2604
}

class YOLOVideoApp(QWidget):
    def __init__(self, model_path, skip_frames=5):
        super().__init__()
        self.model = YOLO(model_path)
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_time = time.time()

        self.queue = queue.Queue(maxsize=10)
        self.running = False
        self.paused = False
        self.video_path = None

        # Shared variable for detections
        self.detections = []  # List of detected objects

        self.setWindowTitle("YOLO Video App")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Video feed label with fixed size
        self.image_label = QLabel("Video Feed")
        self.image_label.setFixedSize(640, 360)
        self.layout.addWidget(self.image_label)

        # System stats labels
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.ram_label = QLabel("RAM Usage: 0%")
        self.fps_label = QLabel("FPS: 0")
        self.layout.addWidget(self.cpu_label)
        self.layout.addWidget(self.ram_label)
        self.layout.addWidget(self.fps_label)

        # Buttons
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.upload_video_dialog)
        self.layout.addWidget(self.upload_button)

        self.start_button = QPushButton("Start Video")
        self.start_button.clicked.connect(self.start_video_task)
        self.layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause/Resume Video")
        self.pause_button.clicked.connect(self.toggle_pause_task)
        self.layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop Video")
        self.stop_button.clicked.connect(self.stop_video_task)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        # Timer for system stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

    def upload_video_dialog(self):
        """Open a file dialog to select a video."""
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if self.video_path:
            self.image_label.setText(f"Loaded: {self.video_path}")
            print(f"Video path: {self.video_path}")

    def start_video_task(self):
        """Start video processing."""
        if not self.running:
            threading.Thread(target=self.start_video, daemon=True).start()

    def toggle_pause_task(self):
        """Pause or resume video processing."""
        self.paused = not self.paused

    def stop_video_task(self):
        """Stop video processing."""
        self.running = False
        self.paused = False
        self.image_label.clear()
        self.image_label.setText("Video Feed")

    def receive_frames(self):
        """Thread function to read frames from the uploaded video."""
        cap = cv2.VideoCapture(self.video_path)
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if ret and not self.queue.full():
                self.queue.put(frame)
            elif not ret:
                break

        cap.release()

    def display_frames(self):
        """Thread function to display and process frames."""
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()

                # Skip frames
                self.frame_counter += 1
                if self.frame_counter % self.skip_frames != 0:
                    continue

                # YOLO detection
                results = self.model(frame)
                self.detections = []
                for result in results:
                    for box in result.boxes:
                        cls = result.names[int(box.cls)]
                        confidence = float(box.conf)
                        self.detections.append({"label": cls, "confidence": confidence})

                # Annotate frame
                detection_frame = results[0].plot()

                # Convert frame to QImage for display
                rgb_image = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(640, 360, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)

    def start_video(self):
        """Start video processing threads."""
        if not self.video_path:
            print("No video selected. Please upload a video.")
            return

        self.running = True
        self.paused = False
        self.receive_thread = threading.Thread(target=self.receive_frames, daemon=True)
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.receive_thread.start()
        self.display_thread.start()

    def update_stats(self):
        """Update CPU and RAM usage stats."""
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        self.cpu_label.setText(f"CPU Usage: {cpu_usage}%")
        self.ram_label.setText(f"RAM Usage: {ram_usage}%")

if __name__ == "__main__":
    # Initialize the PySide6 application
    app = QApplication(sys.argv)
    yolo_app = YOLOVideoApp("best.pt")
    yolo_app.show()

    # Run the PySide6 application
    sys.exit(app.exec())
