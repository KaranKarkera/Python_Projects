# Road Detection using Binary Segmentation with U-Net

This project performs binary segmentation to detect roads in urban environments using the Cityscapes dataset. The pipeline includes preprocessing, training a U-Net model, and inference on both validation and custom images.

---

## 📦 Dependencies

Install all required libraries with:

```bash
pip install numpy torch torchvision accelerate torchinfo pillow pandas matplotlib seaborn
```

---

## 📁 Project Structure

```
.
├── cityscapes_data/
│   ├── train/
│   └── val/
├── road_segmentation_unet.pth
├── script.py
└── README.md
```

* `train/` and `val/`: contain Cityscapes dataset images with input + segmentation mask side-by-side.
* `road_segmentation_unet.pth`: trained model weights.

---

## ⚙️ Configuration

Edit hyperparameters and settings in the `CONFIG` class:

```python
class CONFIG:
    USE_MIXED_PRECISION = "fp16"
    DOWNSCALE = None
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    BATCH_SIZE = 8
    SINGLE_NETWORK_TRAINING_EPOCHS = 15
```

---

## 🧹 Preprocessing

Each Cityscapes image contains:

* **Left half**: Input RGB image
* **Right half**: Segmentation mask

The code extracts a binary mask where:

* `1` indicates the road class
* `0` indicates all other classes

Parallel preprocessing is performed using multiprocessing for speed.

---

## 🧠 Model Architecture: U-Net

A custom U-Net is implemented from scratch, suitable for binary segmentation:

* Contracting Path: 4 downsampling blocks
* Expanding Path: 4 upsampling blocks
* Final 1x1 conv layer outputs a single-channel logits map

---

## 🏋️ Training

Loss functions:

* **BCEWithLogitsLoss** (Binary Cross-Entropy)
* **Dice Loss**

Training metrics:

* Loss
* Dice Coefficient
* IoU (Intersection over Union)

```python
history = train_model(
    model,
    device,
    train_dataloader,
    val_dataloader,
    epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,
    lr=1e-4
)
```

Model is saved at the end:

```python
torch.save(model.state_dict(), "road_segmentation_unet.pth")
```

---

## 📊 Evaluation

Validation metrics are computed after every epoch. You can visualize predictions on:

* Validation batches (`show_inference()`)
* Custom images (`run_inference_on_custom_images()`)

---

## 🖼️ Inference on Custom Images

To run inference on your own images:

1. Prepare RGB images
2. Resize internally to 256x256
3. Use:

```python
custom_image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
run_inference_on_custom_images(model, custom_image_paths, device=device)
```

---

## 📈 Visualization

* Random samples of training images are plotted.
* For each inference:

  * Input image
  * Ground truth (if available)
  * Predicted binary mask

---

## 🔋 Accelerator Support

Mixed precision training and device management are handled using 🤗 `Accelerate`.

---

## 📝 Notes

* The Cityscapes dataset must be preprocessed such that each image contains the RGB input and its corresponding segmentation mask side-by-side.
* Road class is extracted by color matching and mapped using Euclidean distance.
* Adjust `DOWNSCALE` in config if needed to resize images for training speed/memory.

---

## 📬 Contact

For issues or questions, please contact the repository maintainer or open an issue.

---

## 📜 License

This project is released under the MIT License.

---

Happy Training! 🚗🛣️
