# 60 Machine Learning Algorithms Guidebook

## 0.1. Purpose of This Reference

This reference provides a structured introduction to 60 machine learning algorithms and neural network architectures implemented using three major Python frameworks:

- **scikit-learn:** 40 classical machine learning algorithms.
- **PyTorch:** 10 neural network architectures and implementation patterns.
- **TensorFlow/Keras:** 10 neural network architectures and implementation patterns.

The purpose of this reference is to establish a practical and theoretical understanding of machine learning models. It is not intended to serve as a collection of coding examples alone.

For each model, the main objective is to understand:

1. The problem the model is designed to solve.
2. The mathematical principles underlying its learning mechanism.
3. The assumptions made by the model.
4. The characteristics and requirements of the input data.
5. The preprocessing procedures required before training.
6. The hyperparameters that influence model behavior.
7. The appropriate evaluation methods.
8. The model's advantages, limitations, and potential applications.

The central principle is that an algorithm should be selected according to the structure of the problem and the characteristics of the available data, rather than its popularity or implementation convenience.

A model with a sophisticated architecture is not necessarily better than a simpler statistical model. Its suitability must be demonstrated through appropriate experiments and valid evaluation procedures.

---

## 0.2. The Three Frameworks

Although scikit-learn, PyTorch, and TensorFlow can all be used to construct machine learning systems, their design philosophies and primary applications differ.

| Framework | Primary Purpose | Typical Applications |
|---|---|---|
| scikit-learn | Classical machine learning and statistical modeling | Regression, classification, clustering, dimensionality reduction |
| PyTorch | Flexible neural network development and research | Deep learning, computer vision, sequence modeling, generative models |
| TensorFlow/Keras | Neural network development, training, and deployment | Classification, regression, computer vision, sequence modeling |

### scikit-learn

scikit-learn provides standardized implementations of classical machine learning algorithms.

Most estimators follow a consistent interface:

```python
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Its standardized interface makes it particularly suitable for experimentation, model comparison, preprocessing, and conventional predictive modeling.

### PyTorch

PyTorch is a tensor-based deep learning framework that provides automatic differentiation and flexible neural network construction.

A typical training process includes:

- Defining a neural network architecture.
- Preparing input tensors.
- Selecting a loss function.
- Defining an optimizer.
- Performing forward propagation.
- Calculating the loss.
- Computing gradients through backpropagation.
- Updating model parameters.

PyTorch is particularly useful when implementing custom architectures or conducting deep learning research.

### TensorFlow/Keras

TensorFlow provides numerical computation and automatic differentiation infrastructure, while Keras provides high-level interfaces for constructing and training neural networks.

Keras supports two major architectural approaches:

- Sequential API: for straightforward, sequentially connected layers.
- Functional API: for more complex architectures involving multiple inputs, outputs, or non-linear connections.

Both PyTorch and TensorFlow can implement the same neural network architecture. Their primary differences concern the programming interface, training workflow, ecosystem, and implementation details.

**Important distinction:** The 20 deep learning entries in this reference are not necessarily 20 mathematically distinct algorithms. Some represent the same neural network architecture implemented in two different frameworks.

---

## 0.3. The Main Categories of Machine Learning

Before selecting a model, the underlying learning problem must be identified.

### A. Supervised Learning

Supervised learning uses labeled observations. Each training observation consists of input features and a corresponding target value.

The dataset can be represented as:

$$
D = \{(x_i,y_i)\}_{i=1}^{n}
$$

Where:

- $x_i$ represents the input features.
- $y_i$ represents the corresponding target.
- $n$ represents the number of observations.

The objective is to estimate a function:

$$
f:X\rightarrow Y
$$

The model learns the relationship between input features and target values so that it can make predictions for previously unseen observations.

Supervised learning is divided into two principal problem types.

**1. Regression**

Regression predicts continuous numerical values.

Examples:

- Predicting house prices.
- Forecasting financial variables.
- Estimating customer demand.
- Predicting temperature.
- Estimating production costs.

Common algorithms:

- Linear Regression.
- Ridge Regression.
- Lasso Regression.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- Support Vector Regression.

**2. Classification**

Classification predicts discrete categories or classes.

Examples:

- Fraudulent versus legitimate transactions.
- Customer churn versus customer retention.
- Disease-positive versus disease-negative observations.
- Image classification.
- Spam versus non-spam email.

Common algorithms:

- Logistic Regression.
- K-Nearest Neighbors Classifier.
- Support Vector Classifier.
- Decision Tree Classifier.
- Random Forest Classifier.
- Neural Networks.

### B. Unsupervised Learning

Unsupervised learning operates on data without predefined target labels.

The objective is to identify hidden structures, patterns, similarities, or unusual observations within the dataset.

The dataset can be represented as:

$$
D=\{x_i\}_{i=1}^{n}
$$

Common applications include:

- Customer segmentation.
- Market regime identification.
- Pattern discovery.
- Feature compression.
- Anomaly detection.
- Exploratory data analysis.

Common algorithms:

- K-Means.
- DBSCAN.
- Agglomerative Clustering.
- Spectral Clustering.
- Gaussian Mixture Models.
- PCA.
- Isolation Forest.

### C. Deep Learning

Deep learning is a subset of machine learning based on neural networks containing multiple computational layers.

A general neural network layer can be expressed as:

$$
h^{(l)}=f(W^{(l)}h^{(l-1)}+b^{(l)})
$$

Where:

- $h^{(l)}$ represents the output of layer $l$.
- $W^{(l)}$ represents the trainable weights.
- $b^{(l)}$ represents the bias.
- $f$ represents an activation function.

Deep learning models learn their parameters through optimization, typically using backpropagation and gradient-based optimization.

They are particularly useful when dealing with complex, high-dimensional data such as images, audio, text, and sequential observations.

---

## 0.4. Understanding the Dataset

The dataset determines which algorithms are appropriate. Selecting a model before understanding the data can lead to incorrect assumptions, unnecessary computational costs, and misleading results.

### 0.4.1. Basic Dataset Structure

For most traditional machine learning problems, the feature matrix is represented as:

$$
X\in\mathbb{R}^{n\times p}
$$

Where:

- $n$ is the number of observations.
- $p$ is the number of input features.
- Each row represents one observation.
- Each column represents one feature.

For supervised learning, an additional target vector is required:

$$
y\in\mathbb{R}^{n}
$$

For classification, the target values represent categories rather than necessarily continuous numerical values.

Example:

| Age | Income | Transaction Count | Customer Status |
|---:|---:|---:|---|
| 24 | 5000 | 12 | Active |
| 35 | 7200 | 21 | Inactive |
| 41 | 8500 | 17 | Active |
| 29 | 6100 | 9 | Active |

In this example:

- Age, Income, and Transaction Count are input features.
- Customer Status is the target variable.
- The learning problem is classification.

The target must not accidentally be included in the input features. Doing so introduces target leakage and invalidates the evaluation.

### 0.4.2. Types of Data

Different data structures require different preprocessing procedures.

| Data Type | Characteristics | Common Preprocessing |
|---|---|---|
| Numerical | Continuous or discrete numerical values | Imputation, scaling, outlier analysis |
| Categorical | Values representing categories | One-hot encoding or other appropriate encoding |
| Binary | Two possible values | Binary encoding or numerical representation |
| Text | Words, sentences, or documents | Tokenization, TF-IDF, embeddings |
| Image | Pixel-based observations | Resizing, normalization, augmentation |
| Time Series | Temporally ordered observations | Chronological splitting, lag features |
| Sequential Data | Ordered events or tokens | Tokenization, padding, embeddings, masking |

Most classical scikit-learn estimators require numerical feature representations.

Text, images, and sequences generally need to be transformed into appropriate numerical representations before they can be used by these estimators.

### 0.4.3. Missing Values

Missing observations can prevent model training or introduce bias.

Common approaches include:

**Mean Imputation**

Replace missing numerical values with the arithmetic mean:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

This approach is generally appropriate when the mean provides a reasonable representation of the feature.

**Median Imputation**

Replace missing values with the median:

$$
x_{\text{missing}}=\operatorname{Median}(X)
$$

This is often preferable for numerical features with skewed distributions or extreme observations.

**Mode Imputation**

Replace missing categorical values with the most frequent category.

**Model-Based Imputation**

Estimate missing values using relationships between other variables.

This approach can be useful when simpler methods are insufficient, but it introduces additional modeling assumptions.

The imputation strategy must be learned from the training data rather than the entire dataset.

### 0.4.4. Feature Scaling

Feature scaling is particularly important for algorithms that rely on distances, gradients, or regularized coefficients.

**Standardization**

$$
z=\frac{x-\mu}{\sigma}
$$

Where:

- $\mu$ represents the feature mean.
- $\sigma$ represents the feature standard deviation.

Standardization transforms a feature to have approximately zero mean and unit variance.

**Min-Max Normalization**

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

This transformation commonly scales values to the interval $[0,1]$.

Models that commonly benefit from scaling include:

- Logistic Regression.
- Ridge Regression.
- Lasso Regression.
- Support Vector Machines.
- K-Nearest Neighbors.
- Neural Networks.
- K-Means.
- PCA.

Decision Trees and Random Forests are generally much less sensitive to feature scaling because their splits are based on feature thresholds.

Scaling parameters must be estimated from the training set only.

### 0.4.5. Outliers and Data Distribution

Outliers can influence different algorithms in different ways.

Examples:

- Linear Regression can be influenced by extreme observations.
- K-Means uses squared Euclidean distances and can be sensitive to extreme values.
- KNN can be distorted by poorly scaled features or extreme observations.
- Decision Trees are less sensitive to feature scaling but can still be affected by mislabeled observations and unusual distributions.

Useful diagnostic methods include:

- Histograms.
- Box plots.
- Scatter plots.
- Correlation matrices.
- Quantile analysis.
- Distribution comparisons.

An outlier should not automatically be removed.

It may represent:

1. A data collection error.
2. A rare but legitimate observation.
3. An unusual event that the model is specifically expected to detect.

The decision to remove an observation must be supported by an understanding of the data and the research objective.

---

## 0.5. Splitting the Dataset

A supervised machine learning dataset should generally be separated into training, validation, and testing sets.

### Training Set

Used to estimate the model parameters.

### Validation Set

Used to compare models, tune hyperparameters, and make model-selection decisions.

### Test Set

Used for the final evaluation of model generalization.

A possible initial split is:

- 70% training.
- 15% validation.
- 15% testing.

An alternative is an 80/20 training-test split combined with cross-validation on the training set.

The appropriate split depends on dataset size, structure, and the research objective.

**Important considerations:**

- For classification, stratified splitting can preserve class proportions.
- For time-series data, use chronological splitting rather than randomly shuffling observations.
- For repeated observations from the same individual or organization, consider group-based splitting.
- For forecasting, never use future information to predict an earlier observation.

Random splitting is not universally appropriate.

---

## 0.6. Mathematical Foundations

A working understanding of mathematics is essential for interpreting machine learning behavior and understanding why particular models succeed or fail.

### 0.6.1. Linear Algebra

Linear algebra explains how machine learning models represent and transform numerical information.

A linear transformation can be expressed as:

$$
Z=XW+b
$$

Where:

- $X$ is the input matrix.
- $W$ is a matrix of model parameters.
- $b$ is a bias term.
- $Z$ is the resulting representation.

This operation appears in:

- Linear Regression.
- Logistic Regression.
- Neural Networks.
- PCA.
- LDA.
- Singular Value Decomposition.

Important topics include:

- Vectors and matrices.
- Matrix multiplication.
- Matrix rank.
- Eigenvalues and eigenvectors.
- Singular Value Decomposition.
- Orthogonality.
- Covariance matrices.

### 0.6.2. Calculus and Optimization

Machine learning models frequently learn by minimizing an objective function.

A general optimization problem is:

$$
\theta^*=\arg\min_{\theta}J(\theta)
$$

Where:

- $\theta$ represents model parameters.
- $J(\theta)$ represents the objective or loss function.
- $\theta^*$ represents the optimized parameters.

Gradient Descent updates the parameters according to:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}J(\theta_t)
$$

Where $\eta$ represents the learning rate.

This principle is fundamental to neural network training and many iterative optimization methods.

### 0.6.3. Probability and Statistics

Probability provides the theoretical foundation for probabilistic machine learning models and statistical inference.

Important concepts include:

- Conditional probability.
- Bayes' theorem.
- Probability distributions.
- Maximum likelihood estimation.
- Expectation and variance.
- Covariance and correlation.
- Statistical independence.

Bayes' theorem is expressed as:

$$
P(A\mid B)=
\frac{P(B\mid A)P(A)}{P(B)}
$$

This theorem forms the foundation of Naive Bayes classification.

### 0.6.4. Loss Functions

A loss function quantifies the difference between the model's predictions and the actual observations.

**Mean Squared Error**

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

**Mean Absolute Error**

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

**Binary Cross-Entropy**

$$
L=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

**Categorical Cross-Entropy**

$$
L=
-\sum_{k=1}^{C}y_k\log(p_k)
$$

The appropriate loss function depends on the task, target representation, and model output.

### 0.6.5. Regularization

Regularization penalizes model complexity to reduce overfitting.

**L1 Regularization**

$$
J(\beta)=
MSE+\lambda\sum_j|\beta_j|
$$

**L2 Regularization**

$$
J(\beta)=
MSE+\lambda\sum_j\beta_j^2
$$

L1 regularization can force certain coefficients to become exactly zero, while L2 regularization discourages excessively large coefficients.

The regularization strength $\lambda$ must be selected using an appropriate validation procedure.

---

## 0.7. General Model Selection Principles

An algorithm should be selected according to the characteristics of the dataset and the objective of the experiment.

The following are useful starting points:

| Problem | Candidate Models | Common Evaluation Metrics |
|---|---|---|
| Tabular Regression | Linear Regression, Ridge, Random Forest, Gradient Boosting | MAE, RMSE, R² |
| Tabular Classification | Logistic Regression, Random Forest, SVM, Gradient Boosting | F1-score, Precision, Recall, ROC-AUC |
| Clustering | K-Means, DBSCAN, Agglomerative Clustering | Silhouette Score, domain validation |
| Dimensionality Reduction | PCA, Truncated SVD | Explained Variance, Downstream Performance |
| Anomaly Detection | Isolation Forest, LOF, One-Class SVM | Precision, Recall when labels exist |
| Image Data | CNN, Pretrained Vision Models | Accuracy, F1-score, Confusion Matrix |
| Sequential Data | RNN, LSTM, GRU, Transformer | Metrics appropriate to the prediction target |
| Text Classification | Naive Bayes, Logistic Regression, Neural Networks, Transformers | F1-score, Precision, Recall, PR-AUC |

There is no universally optimal machine learning algorithm.

The same algorithm can perform very differently across datasets, even when the datasets belong to the same general problem category.

Model selection must therefore be treated as an empirical process supported by mathematical reasoning and appropriate validation.

---

## 0.8. Recommended Learning Resources

The following resources provide further theoretical explanations, mathematical derivations, implementation details, and practical examples.

### Official Framework Documentation

**1. scikit-learn User Guide**

https://scikit-learn.org/stable/user_guide.html

Primary reference for the 40 classical machine learning implementations.

Recommended topics:

- Supervised Learning.
- Linear Models.
- Support Vector Machines.
- Nearest Neighbors.
- Decision Trees.
- Ensemble Methods.
- Naive Bayes.
- Clustering.
- Decomposition.
- Outlier Detection.

**2. PyTorch Tutorials**

https://docs.pytorch.org/tutorials/

Recommended topics:

- Tensors.
- Datasets and DataLoaders.
- Building Models.
- Automatic Differentiation.
- Optimization.
- Saving and Loading Models.

**3. Keras Guides**

https://keras.io/guides/

Recommended topics:

- Sequential API.
- Functional API.
- Custom Training Loops.
- Convolutional Layers.
- Recurrent Layers.
- Model Evaluation.
- Model Saving.

**4. Keras Examples**

https://keras.io/examples/

Useful for connecting theoretical architectures with practical implementations.

### Mathematical and Theoretical Resources

**5. An Introduction to Statistical Learning**

https://www.statlearning.com/

A recommended reference for classical statistical learning.

Topics include:

- Regression.
- Classification.
- Resampling.
- Regularization.
- Tree-Based Methods.
- Support Vector Machines.
- Deep Learning.
- Unsupervised Learning.

**6. The Elements of Statistical Learning**

https://hastie.su.domains/ElemStatLearn/

A more mathematically detailed reference covering statistical learning theory and advanced predictive modeling.

**7. Dive into Deep Learning**

https://d2l.ai/

An interactive textbook combining mathematical derivations, code, and practical deep learning implementations.

Recommended topics:

- Linear Neural Networks.
- Multilayer Perceptrons.
- Backpropagation.
- CNNs.
- RNNs.
- LSTMs.
- GRUs.
- Attention Mechanisms.
- Transformers.
- GANs.

### Research and Practical Experimentation

**8. Papers with Code**

https://paperswithcode.com/

Useful for connecting research papers with implementations, datasets, and benchmark results.

**9. Google Colab**

https://colab.research.google.com/

A cloud-based notebook environment for running Python experiments and deep learning workloads.

**10. Kaggle**

https://www.kaggle.com/

A resource for finding datasets, examining existing machine learning notebooks, and conducting practical experiments.

---

## 0.9. Recommended Learning Strategy
For each algorithm, follow the same learning procedure:

1. Understand the problem it is designed to solve.
2. Study its mathematical formulation.
3. Identify its assumptions.
4. Understand its data requirements.
5. Implement the model.
6. Evaluate it against a suitable baseline.
7. Investigate its limitations.
8. Experiment with its hyperparameters.
9. Examine how performance changes under different data conditions.
10. Document the experimental results.

The objective is not simply to accumulate implementations. It is to understand why a model works, when it fails, and what its mathematical assumptions imply for real-world data.

---

## 0.10. Final Remarks

Machine learning is not primarily a competition to identify the most sophisticated algorithm.

It is a process of discovering, representing, and approximating relationships in data under specific assumptions and constraints.

A successful experiment requires more than a high evaluation score. It requires a defensible dataset, an appropriate mathematical model, a valid experimental design, careful evaluation, and an understanding of the limitations of the resulting predictions.

Classical machine learning remains highly relevant, particularly for structured data and problems where computational efficiency and interpretability matter.

Deep learning becomes increasingly useful when the data contains complex structures that are difficult to represent through manually engineered features, particularly in computer vision, natural language processing, audio, and sequential modeling.

Both approaches should be understood as complementary tools.
