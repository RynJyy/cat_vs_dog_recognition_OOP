# 🐶 Cats vs 🐱 Dogs Classifier (with GUI)
---

## 📌 About

This project is a Python-based image classification system that distinguishes between cats and dogs using transfer learning with MobileNetV2. It includes two main components:

- A training pipeline that prepares, augments, and trains a model using TensorFlow and Keras.
- A graphical user interface (GUI) built with Tkinter, allowing users to upload an image and receive a real-time prediction.
- Designed for both educational and practical use, the codebase demonstrates the four pillars of Object-Oriented Programming (OOP)—encapsulation, abstraction, inheritance, and polymorphism—through clear class structures and modular design.

Whether you're training your own model or simply exploring how machine learning integrates with GUI applications, this project offers a full end-to-end example.

---

## 📁 Project Structure

```
archive/
└── dogs_vs_cats/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/

main_train_model.py
image_classify.py
cat_dog_classify.py
dogs_vs_cats_transfer.h5 (generated after training)
cat_dog_classify_app.py
main_app_gui.py
background.jpg
```

---

## 🚀 Features

- ✅ Transfer learning with MobileNetV2 for high accuracy
- ✅ Data augmentation for improved generalization
- ✅ Clean and colorful GUI to classify images
- ✅ Clear separation of model training and GUI code
- ✅ Interactive buttons: Upload, Clear, Exit

---

## 📦 Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pillow
- Tkinter (usually included with Python)

Install dependencies:

```bash
pip install tensorflow numpy pillow
```

---

## 🧠 Model Training

Train the model on your dataset:

```bash
python main_train_model.py
```

This will save the model as `dogs_vs_cats_transfer.h5`.

---

## 💻 Run the GUI App

After training, launch the classifier GUI:

```bash
python main_app_gui.py
```

Upload any image of a cat or dog and see the prediction!

---

## 👨‍🏫 OOP Principles Demonstrated

1. Encapsulation
   - Each class (ImageClassifier, CatDogClassifier, and CatDogClassifierGUI) encapsulates its data and methods.

2. Abstraction
   - Complex processes like training, loading data, or GUI interactions are abstracted into simple method calls.

3. Inheritance
   - CatDogClassifier inherits from ImageClassifier to extend functionality while reusing core logic.

4. Polymorphism
   - Methods like load_data() or build_model() behave differently depending on the subclass implementing them.

---

## 🖼️ GUI Preview

> Includes a background image, stylized labels, and result display. You can customize the `background.jpg` file.

---

## 🛠 How It Works

- Uses **MobileNetV2** pretrained on ImageNet (transfer learning)
- Adds a global pooling layer + dense layers on top
- Uses `ImageDataGenerator` for training and validation
- GUI uses `tkinter`, `PIL`, and `filedialog` to display and classify user images

---

## 📌 Inspiration and Credits

- Inspired by [leemengtw/cat-recognition-train](https://github.com/leemengtw/cat-recognition-train) — A great example of how to apply transfer learning for image classification.

---

## ✅ To Do

- [ ] Add real-time webcam prediction (optional)
- [ ] Add model performance charts (accuracy/loss)
- [ ] Allow batch predictions from a folder

---

## 👨‍💻 Author

**Ryan Canlas**

---

## 📄 License

This project is licensed under the **MIT License**.
<details>
<summary>Click to expand MIT License</summary>

```
MIT License

Copyright (c) 2025 Ryan Canlas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
</details>
