# 🍎 Food Freshness Checker

> A smart web app that uses **Machine Learning (TensorFlow.js + MobileNet)** to detect whether food is **fresh** or **rotten** based on image uploads.  
> Built with **React**, **Bootstrap**, and a clean **black & red aesthetic** theme — fully deployable on **Vercel**.

---

## 🌐 Live Demo
🔗 **[View on Vercel](https://your-vercel-link.vercel.app/)**  
*(Replace this with your deployed link once live)*

---

## 🖼️ Screenshot

![Food Freshness Checker UI](./Screenshot%202025-11-08%20110610.png)

---

## 🧾 Project Structure

food-freshness-checker/
├── public/
├── src/
│   ├── FoodFreshnessChecker.jsx  # Main component with ML + UI
│   ├── App.js                    # Root component
│   ├── index.js                  # App entry
│   └── index.css                 # Global CSS
├── package.json
├── README.md
└── .gitignore

---

## 🚀 Features

✅ Upload fresh & rotten food images as references  
✅ Upload a test image to predict food condition  
✅ Uses **TensorFlow.js MobileNet embeddings** for ML-based comparison  
✅ Includes a **color heuristic** to detect browning or spoilage  
✅ Fully client-side (no backend required)  
✅ Responsive and professionally designed with **Bootstrap**  
✅ Deploy easily with **Vercel**

---

## 🧠 How It Works

1. **Upload a fresh food image** → used as a visual reference.  
2. *(Optionally)* **Upload a rotten image** → helps the model understand what “bad” looks like.  
3. **Upload a test image** → the app analyzes similarity between reference and test images using:
   - MobileNet embeddings (for visual features)
   - Cosine similarity (for comparing image features)
   - Color analysis (to check for dark/brown tones)
4. Combines all results to predict:  
   🟢 **EDIBLE** or 🔴 **NOT EDIBLE**

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | React (Create React App) |
| Styling | Bootstrap 5 + Custom CSS |
| Machine Learning | TensorFlow.js + MobileNet |
| Hosting | Vercel |
| Language | JavaScript (ES6) |

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/food-freshness-checker.git
cd food-freshness-checker
npm install
