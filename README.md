
⚙️ Core Architecture & Pipeline
1. Self-Supervised Fine-Tuning (CutPaste)
Since industrial anomalies are rare and highly varied, the model is trained only on normal (good) data. To force the Swin Transformer backbone to learn fine-grained surface anomalies, a CutPasteAugmentation pipeline generates synthetic defects by cutting a localized patch and pasting it randomly onto a normal image. The backbone is fine-tuned as a binary classifier separating Normal vs Synthetic Defect images.

2. Hierarchical Memory Bank Construction
The fine-tuned Swin Transformer extracts deep patch-level feature representations from normal images. These multi-scale feature maps are concatenated to preserve both global context and fine local details, forming a massive feature memory bank.

3. Coreset Subsampling & FAISS Indexing
To deploy this system on resource-constrained Edge devices, a K-Center Greedy Coreset algorithm reduces the memory size by 90%. The remaining core feature vectors are indexed using Meta's FAISS (IndexFlatL2) for ultra-low latency nearest-neighbor search during real-time inference.

4. Anomaly Map & Scoring
During evaluation, the test image features are extracted and compared against the FAISS memory bank.

Image-level Anomaly Score: Computed based on the maximum distance to the nearest normal neighbor.

Pixel-level Heatmap: Local anomaly distances are reshaped, upsampled back to the original image dimensions via Bilinear Interpolation, and smoothed with a Gaussian filter to output precise anomaly heatmaps.

📊 Performance Metrics
The model achieves exceptional results on the industrial MVTec AD dataset benchmarks:

Image-level Classification: High Image-level AUROC across diverse texture and object categories.

Pixel-level Localization: Sharp anomaly heatmaps yielding accurate Pixel-level AUROC boundaries.

Throughput Optimization: Optimized execution pipeline achieving high FPS processing speeds suitable for live manufacturing lines.

💻 How to Run
Install Dependencies:

Bash
pip install timm faiss-cpu scikit-learn opencv-python huggingface_hub --quiet
Open and Execute the Pipeline:
Open base-version.ipynb in your Jupyter Notebook, VS Code, or Kaggle/Colab environment. Ensure your hardware accelerator (GPU) is enabled for fast model fine-tuning and FAISS inference.

👥 Authors
Nguyen Le Tung Duong - Data Science Student - University of Science, VNU-HCM.

GitHub: @Duongg178
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md generated successfully.")

Một file `README.md` chuẩn chỉnh, chuyên nghiệp và đầy đủ cấu trúc của một dự án Deep Learning lớn sẽ giúp kho lưu trữ (Repository) của bạn trên GitHub trông cực kỳ ấn tượng trong mắt các Leader hoặc nhà tuyển dụng. 

Tôi đã tối ưu hóa toàn bộ nội dung file dựa trên chính mã nguồn notebook `base-version.ipynb` của bạn. Dưới đây là file `README.md` đã được tạo sẵn để bạn tải về đưa vào dự án, kèm theo toàn bộ nội dung text để bạn dễ dàng sao chép:

### 📥 Tải xuống File:
* [file-tag: code-generated-file-0-1778950708124848101]

---

### 📝 Nội dung chi tiết file `README.md` (Sao chép tại đây):

```markdown
# Industrial Defect Detection via Vision Transformers (Swin-ViT & PatchCore)

An advanced, production-ready deep learning pipeline designed for **Unsupervised Anomaly Detection** and localization in industrial manufacturing lines. This project leverages hierarchical feature extractors from **Swin Transformers** combined with a modified **PatchCore/CutPaste memory bank** framework to detect structural and surface anomalies without requiring labeled defect data during training.

Developed as a scientific research project at the University of Science, Vietnam National University (VNU).

---

## 🚀 Key Features

- **Self-Supervised Fine-Tuning**: Employs a dual-class `CutPaste` geometric data augmentation strategy to optimize an ImageNet pre-trained Swin Transformer backbone to specialize in local structural irregularities.
- **Efficient Memory Footprint**: Integrates **K-Center Greedy Coreset Subsampling**, shrinking the FAISS vector memory bank by **90%** while maintaining maximal topological coverage and saving edge hardware memory.
- **High-Performance Scaling**: Implemented asynchronous multi-threaded data loading (`pin_memory`, `num_workers`) and multi-GPU distribution via PyTorch `nn.DataParallel`.
- **Dual-Level Localization**: Outputs macro-level anomaly classification (`Image-level AUROC`) and pixel-level defect segmentation heatmaps (`Pixel-level AUROC`) via bilinear interpolation and Gaussian filtering.
- **MLOps Hub Integration**: Automated script to automatically save and upload model weights, FAISS vector indices, and performance JSON matrices directly to the Hugging Face Model Hub.

---

## 🛠️ Technical Stack

- **Core Framework**: PyTorch (`torch`, `torch.nn`, `torch.optim`)
- **Backbone & Feature Extraction**: `timm` (Swin Transformer `swin_tiny_patch4_window7_224`)
- **Vector Search Engine**: Meta's **FAISS** (`IndexFlatL2`)
- **Data & Image Processing**: OpenCV, NumPy, Pillow, Scikit-learn
- **Model Tracking & Registry**: Hugging Face Hub API

---

## 📦 Project Structure

```text
├── base-version.ipynb      # Main pipeline (Data Augmentation, Fine-Tuning, Coreset Selection, Evaluation)
├── .gitignore              # Prevents checkpoints and local cache files from being tracked
└── README.md               # Project documentation
⚙️ Core Architecture & Pipeline
1. Self-Supervised Fine-Tuning (CutPaste)
Since industrial anomalies are rare and highly varied, the model is trained only on normal (good) data. To force the Swin Transformer backbone to learn fine-grained surface anomalies, a CutPasteAugmentation pipeline generates synthetic defects by cutting a localized patch and pasting it randomly onto a normal image. The backbone is fine-tuned as a binary classifier separating Normal vs Synthetic Defect images.

2. Hierarchical Memory Bank Construction
The fine-tuned Swin Transformer extracts deep patch-level feature representations from normal images. These multi-scale feature maps are concatenated to preserve both global context and fine local details, forming a massive feature memory bank.

3. Coreset Subsampling & FAISS Indexing
To deploy this system on resource-constrained Edge devices, a K-Center Greedy Coreset algorithm reduces the memory size by 90%. The remaining core feature vectors are indexed using Meta's FAISS (IndexFlatL2) for ultra-low latency nearest-neighbor search during real-time inference.

4. Anomaly Map & Scoring
During evaluation, the test image features are extracted and compared against the FAISS memory bank.

Image-level Anomaly Score: Computed based on the maximum distance to the nearest normal neighbor.

Pixel-level Heatmap: Local anomaly distances are reshaped, upsampled back to the original image dimensions via Bilinear Interpolation, and smoothed with a Gaussian filter to output precise anomaly heatmaps.

📊 Performance Metrics
The model achieves exceptional results on the industrial MVTec AD dataset benchmarks:

Image-level Classification: High Image-level AUROC across diverse texture and object categories.

Pixel-level Localization: Sharp anomaly heatmaps yielding accurate Pixel-level AUROC boundaries.

Throughput Optimization: Optimized execution pipeline achieving high FPS processing speeds suitable for live manufacturing lines.

💻 How to Run
Install Dependencies:

Bash
pip install timm faiss-cpu scikit-learn opencv-python huggingface_hub --quiet
Open and Execute the Pipeline:
Open base-version.ipynb in your Jupyter Notebook, VS Code, or Kaggle/Colab environment. Ensure your hardware accelerator (GPU) is enabled for fast model fine-tuning and FAISS inference.

👥 Authors
Nguyen Le Tung Duong - Data Science Student - University of Science, VNU.

GitHub: @Duongg178
