# nh-n-di-n-bi-n-s-xe-
# BÁO CÁO KỸ THUẬT VÀ HƯỚNG DẪN THỰC THI: NHẬN DIỆN BIỂN SỐ XE VỚI YOLOV8 VÀ EASYOCR (IMAGE & VIDEO)

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Hệ thống Nhận diện Biển số xe Tự động (Automatic License Plate Recognition - ALPR) được xây dựng dựa trên kiến trúc hai giai đoạn (Two-stage Pipeline):

1. **Giai đoạn 1 - Phát hiện (Detection):** Sử dụng mô hình **YOLOv8** (You Only Look Once v8) để định vị chính xác vùng chứa biển số xe (Bounding Box) trên khung hình (Hình ảnh tĩnh hoặc Video stream).
2. **Giai đoạn 2 - Tiền xử lý & Nhận diện ký tự (Preprocessing & OCR):** Trích xuất vùng biển số (ROI - Region of Interest), áp dụng các thuật toán xử lý ảnh số (Binarization, Noise Reduction, Perspective Transform) và sử dụng **EasyOCR** để chuyển đổi hình ảnh ký tự thành định dạng chuỗi văn bản (String).

```
+------------------+     +------------------+     +-------------------+     +------------------+
|                  |     |                  |     |                   |     |                  |
|  Input Source    | --> |  YOLOv8 Detector | --> |  Crop & Preprocess| --> |  EasyOCR Engine  |
|  (Image/Video)   |     |  (Find Plate)    |     |  (ROI Filtering)  |     |  (Text Extract)  |
|                  |     |                  |     |                   |     |                  |
+------------------+     +------------------+     +-------------------+     +------------------+
                                                                                  |
                                                                                  v
                                                                        +------------------+
                                                                        |                  |
                                                                        | Output Rendered  |
                                                                        | (BBox + License) |
                                                                        |                  |
                                                                        +------------------+
```

---

## 2. CẤU TRÚC DỰ ÁN (PROJECT STRUCTURE)

```text
alpr_yolov8_ocr/
│
├── weights/
│   └── license_plate_detector.pt   # File trọng số YOLOv8 đã train
│
├── data/
│   ├── inputs/                     # Thư mục chứa ảnh/video đầu vào
│   └── outputs/                    # Thư mục chứa kết quả xuất ra
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Tiền xử lý ảnh & Lọc ký tự chuẩn Việt Nam
│   └── tracker.py                  # Module theo dõi đối tượng cho Video (BYTETracker)
│
├── main.py                         # File thực thi chính (Chạy Ảnh & Video)
├── requirements.txt                # Các thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn sử dụng
```

---

## 3. KHỞI TẠO MÔI TRƯỜNG & PHỤ THUỘC (DEPENDENCIES)

Tạo tập tin `requirements.txt` với nội dung sau:

```text
ultralytics>=8.0.0
easyocr>=1.7.0
opencv-python>=4.8.0
numpy>=1.23.0
torch>=2.0.0
torchvision>=0.15.0
filterpy>=1.4.5
```

Lệnh cài đặt:
```bash
pip install -r requirements.txt
```

---

## 4. TIỀN XỬ LÝ VÀ CHUẨN HÓA KÝ TỰ (SRC/UTILS.PY)

Biển số xe Việt Nam thường gặp các vấn đề về nhiễu, góc nghiêng và đốm sáng. Module `utils.py` đảm nhận việc làm sạch hình ảnh và loại bỏ các ký tự nhiễu sau OCR.

```python
import cv2
import numpy as np
import re

def preprocess_plate(crop_img):
    """
    Tiền xử lý hình ảnh vùng biển số xe nhằm gia tăng độ chính xác cho OCR.
    """
    # 1. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 2. Tăng cường độ tương phản (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # 3. Phóng to ảnh để OCR dễ đọc các đường nét ký tự
    resized = cv2.resize(gray_enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 4. Tẩy nhiễu bằng Blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # 5. Phân ngưỡng nhị phân (Otsu's Thresholding)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def format_license_plate(text):
    """
    Chuẩn hóa chuỗi ký tự thu được từ OCR theo quy chuẩn biển số Việt Nam.
    - Loại bỏ ký tự đặc biệt
    - Chuyển chữ thường thành chữ hoa
    """
    # Chỉ giữ lại chữ cái viết hoa và chữ số
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Sửa lỗi phổ biến của OCR (VD: Nhầm 'O' thành '0', 'I' thành '1' hoặc ngược lại)
    # Tùy thuộc vào vị trí ký tự trên biển số có thể áp dụng Map quy đổi linh hoạt
    
    return clean_text

def validate_plate_format(text):
    """
    Kiểm tra tính hợp lệ sơ bộ của chuỗi biển số xe Việt Nam.
    """
    if len(text) < 7 or len(text) > 10:
        return False
    return True
```

---

## 5. CHƯƠNG TRÌNH CHÍNH (MAIN.PY)

Mã nguồn dưới đây hỗ trợ xử lý cả **Hình ảnh tĩnh (Image)** và **Luồng Video/Camera**.

```python
import cv2
import torch
import numpy as np
import easyocr
from ultralytics import YOLO
import time
from src.utils import preprocess_plate, format_license_plate, validate_plate_format

class ALPRPipeline:
    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        """
        Khởi tạo Pipeline ALPR với YOLOv8 và EasyOCR.
        """
        print("[INFO] Đang khởi tạo mô hình YOLOv8 và EasyOCR...")
        # Kiểm tra GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Tác vụ sử dụng thiết bị tính toán: {self.device}")

        # Khởi tạo YOLOv8
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Khởi tạo EasyOCR (Cấu hình đọc tiếng Anh/Con số)
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))

    def process_frame(self, frame):
        """
        Xử lý từng khung hình: Phát hiện biển số, Cắt ROI, OCR và vẽ Bounding Box.
        """
        # 1. Chạy YOLOv8 Detector
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        for box in results.boxes:
            # Lấy tọa độ Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            # 2. Trích xuất Vùng biển số (Crop ROI)
            # Mở rộng nhẹ viền xung quanh (Padding)
            h, w, _ = frame.shape
            pad = 4
            crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
            crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)

            license_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if license_crop.size == 0:
                continue

            # 3. Tiền xử lý vùng ảnh cắt
            processed_crop = preprocess_plate(license_crop)

            # 4. Chạy EasyOCR trên vùng ảnh cắt
            ocr_results = self.reader.readtext(processed_crop)

            raw_text = ""
            for res in ocr_results:
                text_bbox, text, text_score = res
                if text_score > 0.2:  # Bỏ qua các kết quả có độ tin cậy quá thấp
                    raw_text += text + " "

            # 5. Chuẩn hóa chuỗi văn bản
            formatted_text = format_license_plate(raw_text)

            # 6. Vẽ kết quả lên khung hình chính
            if formatted_text and validate_plate_format(formatted_text):
                # Vẽ khung Bounding Box quanh biển số
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Hiển thị Văn bản nhận diện và Score
                label = f"{formatted_text} ({confidence:.2f})"
                
                # Tính kích thước background chứa text
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def run_image(self, image_path: str, output_path: str = None):
        """
        Thực thi pipeline trên một tệp hình ảnh.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Không thể đọc tệp ảnh: {image_path}")
            return

        processed_frame = self.process_frame(frame)

        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"[SUCCESS] Đã lưu kết quả tại: {output_path}")

        cv2.imshow("ALPR Image Result", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_video(self, video_path: str, output_path: str = None):
        """
        Thực thi pipeline trên tệp Video hoặc Live Camera Stream.
        """
        cap = cv2.VideoCapture(video_path if video_path != '0' else 0)
        if not cap.isOpened():
            print(f"[ERROR] Không thể mở nguồn video/camera: {video_path}")
            return

        # Lấy thông số Video để ghi file output nếu cần
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("[INFO] Bắt đầu xử lý Video Stream. Nhấn 'Q' để thoát.")
        
        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Tính FPS hiển thị
            current_time = time.time()
            fps_display = 1 / (current_time - prev_time)
            prev_time = current_time

            # Xử lý khung hình
            processed_frame = self.process_frame(frame)

            # Vẽ thông số FPS
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if out_writer:
                out_writer.write(processed_frame)

            cv2.imshow("ALPR Video Stream", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Đường dẫn file trọng số (Thay thế bằng file weight bạn trained hoặc tải về)
    MODEL_WEIGHTS = "weights/license_plate_detector.pt" 

    # Khởi tạo Pipeline
    pipeline = ALPRPipeline(model_path=MODEL_WEIGHTS, conf_threshold=0.35)

    # --- CHẠY THỬ VỚI HÌNH ẢNH ---
    # pipeline.run_image(image_path="data/inputs/test_car.jpg", output_path="data/outputs/result_car.jpg")

    # --- CHẠY THỬ VỚI VIDEO ---
    pipeline.run_video(video_path="data/inputs/traffic_video.mp4", output_path="data/outputs/result_video.mp4")

    # --- CHẠY THỬ VỚI WEBCAM REALTIME ---
    # pipeline.run_video(video_path='0')
```

---

## 6. ĐÁNH GIÁ VÀ TỐI ƯU HÓA HIỆU NĂNG (PERFORMANCE OPTIMIZATION)

Khi triển khai thực tế trên môi trường Production hoặc Edge Device (như Nvidia Jetson / Raspberry Pi), bạn nên cân nhắc các phương án tối ưu sau:

1. **Khử trùng lặp chuỗi trên Video Stream (Frame Aggregation & Multi-object Tracking):**
   - Kết hợp thuật toán **BYTETracker** hoặc **SORT** để cấp ID định danh cho từng xe trong Video.
   - Chỉ lưu lại chuỗi OCR có `confidence_score` cao nhất sau $N$ frames liên tiếp của cùng 1 ID xe.

2. **Tăng tốc xử lý Realtime (Inference Acceleration):**
   - **TensorRT / ONNX Export:** Chuyển đổi mô hình PyTorch sang định dạng TensorRT bằng lệnh:
     ```bash
     yolo export model=weights/license_plate_detector.pt format=engine device=0
     ```
   - **Batch Processing:** Nhóm các ảnh cắt ROI lại thành một Batch trước khi đưa vào EasyOCR để tận dụng tối đa sức mạnh tính toán song song của GPU.
