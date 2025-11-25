# Image Sketch & Cartoon Converter

Ứng dụng Python (Tkinter) chuyển ảnh thành:
- Cartoon (quantization + edge detection)
- Pencil sketch (đen trắng, màu)

## 1. Yêu cầu

- Python 3.8+
- pip, venv
- Thư viện:
  - `numpy`
  - `Pillow`
- Tkinter:
  - Windows / macOS: thường có sẵn
  - Ubuntu/Debian: `sudo apt install python3-tk`

## 2. Cài đặt

```bash
cd image_sketch_app

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
