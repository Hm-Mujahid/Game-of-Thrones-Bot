import gensim
import os
import difflib
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

# ======================================================
# Load model
# ======================================================
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "word2vec.model")
    print("Loading Word2Vec model...")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("Please run model.py first to train and save the model.\n")
        exit()

    model = gensim.models.Word2Vec.load(model_path)
    print("✅ Model loaded successfully!\n")
    return model

# ======================================================
# Helper: Auto-correct misspelled words
# ======================================================
def correct_word(model, word):
    word = word.lower().strip()
    vocab = model.wv.key_to_index.keys()
    matches = difflib.get_close_matches(word, vocab, n=1, cutoff=0.75)
    if matches:
        corrected = matches[0]
        if corrected != word:
            print(f"🔹 Auto-corrected '{word}' → '{corrected}'")
        return corrected
    else:
        return word

# ======================================================
# Menu
# ======================================================
def show_menu():
    print("\n📘 Choose an operation to perform:")
    print("1️⃣  Find most similar words")
    print("2️⃣  Find similarity between two words")
    print("3️⃣  Perform word analogy (A is to B as C is to ?)")
    print("4️⃣  Find the odd one out in a list of words")
    print("5️⃣  Show vocabulary size")
    print("6️⃣  Show vector representation of a word")
    print("7️⃣  Plot 2D graphical representation of words")
    print("8️⃣  Exit")

# ======================================================
# Operations
# ======================================================
def most_similar(model):
    word = input("Enter a word: ")
    word = correct_word(model, word)
    if word in model.wv.key_to_index:
        results = model.wv.most_similar(word, topn=10)
        print(f"\nTop 10 words similar to '{word}':")
        for w, score in results:
            print(f"  {w}: {score:.4f}")
    else:
        print(f"'{word}' not found in vocabulary.\n")

def similarity(model):
    w1 = correct_word(model, input("Enter first word: "))
    w2 = correct_word(model, input("Enter second word: "))
    if w1 in model.wv.key_to_index and w2 in model.wv.key_to_index:
        sim = model.wv.similarity(w1, w2)
        print(f"\nSimilarity between '{w1}' and '{w2}': {sim:.4f}")
    else:
        print("One or both words not found in vocabulary.\n")

def analogy(model):
    a = correct_word(model, input("Enter word A: "))
    b = correct_word(model, input("Enter word B: "))
    c = correct_word(model, input("Enter word C: "))
    try:
        result = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
        print(f"\n'{a}' is to '{b}' as '{c}' is to ?")
        for w, score in result:
            print(f"  {w}: {score:.4f}")
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}\n")

def odd_one_out(model):
    words = input("Enter words separated by spaces: ").lower().split()
    words = [correct_word(model, w) for w in words]
    try:
        odd = model.wv.doesnt_match(words)
        print(f"\nOdd one out: {odd}")
    except KeyError:
        print("One or more words not found in vocabulary.\n")

def vocab_size(model):
    print(f"\nVocabulary size: {len(model.wv.index_to_key)} words\n")

def word_vector(model):
    word = correct_word(model, input("Enter a word: "))
    if word in model.wv.key_to_index:
        vector = model.wv[word]
        print(f"\nVector representation for '{word}':\n")
        print(vector)
        print(f"\nVector length: {len(vector)} dimensions")
    else:
        print(f"'{word}' not found in vocabulary.\n")

def plot_words(model):
    words = input("Enter words separated by spaces: ").lower().split()
    words = [correct_word(model, w) for w in words]

    valid_words = [w for w in words if w in model.wv.key_to_index]
    if len(valid_words) < 2:
        print("Need at least 2 valid words for plotting.\n")
        return

    # Get vectors
    vectors = np.array([model.wv[w] for w in valid_words])

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    # Create a scatter plot using Plotly
    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        text=valid_words,
        title="2D Word Embedding Visualization (PCA)",
        labels={"x": "PCA 1", "y": "PCA 2"}
    )
    fig.update_traces(marker=dict(size=12, color="skyblue"), textposition="top center")
    fig.show()

# ======================================================
# Main Loop
# ======================================================
def main():
    model = load_model()

    while True:
        show_menu()
        choice = input("\nEnter your choice (1–8): ")

        if choice == '1':
            most_similar(model)
        elif choice == '2':
            similarity(model)
        elif choice == '3':
            analogy(model)
        elif choice == '4':
            odd_one_out(model)
        elif choice == '5':
            vocab_size(model)
        elif choice == '6':
            word_vector(model)
        elif choice == '7':
            plot_words(model)
        elif choice == '8':
            print("👋 Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()
