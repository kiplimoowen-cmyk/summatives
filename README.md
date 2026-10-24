# summatives
# EcoSort: Intelligent Waste Management Assistant

EcoSort is an end-to-end, multi-modal intelligent assistant developed in partnership with Metro City's waste management department. Designed to optimize residential recycling and minimize sorting friction at municipal facilities, the system processes both image uploads and natural language descriptions to predict waste categories and dynamically retrieve structured local disposal policies.

---

##  Project Architecture & Key Features

```
                     +----------------------+
                     |  Resident Input      |
                     |  (Image or Text)     |
                     +-----------+----------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v [Image Input]                                 v [Text Input]
+------------------------+                      +------------------------+
| MobileNetV2 CNN        |                      | Multinomial Naive Bayes|
| (Feature Extractor)    |                      | (TF-IDF Vectorizer)    |
+-----------+------------+                      +-----------+------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v [Predicted Waste Category]
                        +------------------------+
                        |  Semantic Retrieval    |
                        |  (TF-IDF Cosine Sim)   |
                        +-----------+------------+
                                    |
                                    v [Contextual Policy Match]
                        +------------------------+
                        |  Generative RAG        |
                        |  (FLAN-T5 Generator)   |
                        +-----------+------------+
                                    |
                                    v
                     +----------------------+
                     | Tailored Recycling   |
                     | Instruction Delivery |
                     +----------------------+
```

### 1. Multi-Class Image Classification (CNN)
- **Architecture**: Transfer learning leveraging **MobileNetV2** (pretrained on ImageNet) with a customized classification head (Dropout, Global Average Pooling, and a 9-class Softmax layer).
- **Fine-Tuning**: Selected unfreezing of the top 19 deep layers under a highly constrained, stable learning rate ($10^{-4}$ scaled down to $2 \times 10^{-5}$ via learning rate decay).

### 2. Natural Language Text Classification (NLP)
- **Feature Extraction**: Unigram bag-of-words model using **TF-IDF vectorization**.
- **Classifier**: High-efficiency, low-latency **Multinomial Naive Bayes** classifier mapping user queries and description keywords to one of the 9 waste classes.

### 3. Contextual Document Retrieval & Generative RAG
- **Database**: Localized, structured JSON database containing detailed recycling codes, preparation steps, and constraints.
- **Retrieval Engine**: Sparse semantic retrieval matching user queries to target documents using a TF-IDF vectorizer and cosine similarity.
- **Generative Refinement**: Leverages **FLAN-T5-small** to synthesize clean, concise, and structured instructions, filtering out irrelevant policy text.

---

##  Performance & Evaluation Metrics

###  Deep Learning Vision Pipeline (RealWaste Dataset)
- **Dataset Volume**: 4,752 waste material images spanning 9 classes.
- **Transfer Learning Accuracy**: ~80.43% 
- **Fine-Tuned System Accuracy**: **87.08%**
- **Highest Performing Classes**:
  - **Vegetation**: 1.00 Precision / 0.94 Recall (0.97 F1-Score)
  - **Food Organics**: 0.85 Precision / 0.97 Recall (0.91 F1-Score)
  - **Metal**: 0.93 Precision / 0.89 Recall (0.91 F1-Score)

### Text Classification NLP Pipeline
- **Dataset Volume**: 4,993 clean user descriptions.
- **Accuracy**: **99.60%** (near-perfect classification with only 4 minor word-overlap edge cases across the entire test subset).
- **Macro F1-Score**: 1.00

### Semantic Policy Retrieval
- Successfully retrieved the correct policy documents across all validated target classes with high-confidence cosine similarity margins (e.g., **Paper: 0.902**, **Cardboard: 0.803**).


## Tech Stack & Libraries Used

- **Framework**: `TensorFlow` / `Keras` (Computer Vision model development & training)
- **Transformer SDK**: `Hugging Face Transformers` / `PyTorch` (Seq2Seq text generation with FLAN-T5)
- **Machine Learning**: `Scikit-Learn` (Naive Bayes, TF-IDF vectorization, Cosine Similarity, Metrics evaluation)
- **Data Engineering**: `Pandas`, `NumPy` (Feature preprocessing and serialization management)
- **Visualization**: `Matplotlib`, `Seaborn` (Confusion matrices, performance trends, and learning curves)

---

## Project Structure
```
├── data/
│   ├── realwaste_data/             # RealWaste classification images
│   ├── waste_descriptions.csv       # User text descriptions dataset
│   └── waste_policy_documents.json # Local municipal recycling codes
├── notebooks/
│   └── EcoSort_Assistant.ipynb     # Complete development pipeline
└── README.md                       # Repository Documentation
