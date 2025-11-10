# 🐉 Game of Thrones Word2Vec Project

## 🔥 Explore the world of Westeros using Word2Vec + NLP!

This project uses **Word2Vec** and **Natural Language Processing (NLP)** to understand and analyze text data from the *Game of Thrones* book series.  
You can explore **similar words**, **word analogies**, **semantic relationships**, and even **visualize word embeddings** in 2D using **Plotly**.

---

## 🧠 Features

- 📘 Train a **custom Word2Vec model** on Game of Thrones book text  
- 🔍 Explore:
  - Most similar words  
  - Word similarity score  
  - Word analogies (`A is to B as C is to ?`)  
  - Odd one out in a word list  
  - Word vector representations  
- 🧩 Auto-corrects spelling mistakes in user input  
- 📊 Visualize relationships between words using **Plotly (interactive graphs)**  
- ⚙️ Simple CLI (Command Line Interface) for easy exploration  

---

## 📦 Dataset

The dataset used for this project is **"The Game of Thrones Books"** from Kaggle.

### 🔗 Download Link
📥 [Get the dataset from Kaggle](https://www.kaggle.com/datasets/)

*(Search for “The Game of Thrones Books” on Kaggle if the link changes.)*

### 🗂️ Setup Instructions
1. Download and extract the dataset ZIP file from Kaggle.  
2. Create a folder named **`data`** inside your project directory.  
3. Place all the `.txt` files (the book contents) inside the `data/` folder.

Your project structure should look like this:

got/
├── data/
│ ├── book1.txt
│ ├── book2.txt
│ ├── book3.txt
│ ├── book4.txt
│ └── book5.txt
├── src/
│ ├── preprocess.py
│ ├── operations_menu.py
│ └── word2vec.model
├── .venv
└── README.md


---

## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/got-word2vec.git
   cd got-word2vec
   ```bash
   
