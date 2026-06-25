# Predictive Analytics for Restaurant Expansion

A machine learning project that predicts restaurant franchise profitability based on city population using **Linear Regression with Gradient Descent** implemented from scratch in Python.

## 📌 Project Overview
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has restaurants in various cities, and you have historical data for profits and populations from those cities. This project develops a predictive model to evaluate candidate cities and identify regions that potentially yield higher returns, assisting in strategic data-driven business expansion.

## 📊 Dataset Specifications
The model is trained using a dataset (`ex1data1.txt`) consisting of 97 historical data points:
* **Feature ($x$):** City population (in units of 10,000 people). For example, a value of `6.1101` corresponds to a population of 61,101.
* **Target ($y$):** Average monthly profit of the franchise in that city (in units of $10,000). For example, `17.592` signifies a monthly profit of \$175,920, while `-2.6807` represents a net loss of \$26,807.

## 🛠️ Machine Learning Architecture
Instead of using high-level machine learning libraries like `scikit-learn`, the linear regression algorithm, cost function, and optimization routine are implemented purely using foundational libraries like **NumPy**.

### 1. Model Representation
The linear model predicts the profit $f_{wb}(x^{(i)})$ for a given population $x^{(i)}$ using the linear function:
$$f_{wb}(x^{(i)}) = wx^{(i)} + b$$
Where $w$ is the slope/weight coefficient and $b$ is the y-intercept/bias parameter.

### 2. Cost Function (Mean Squared Error)
To measure how well our model fits the training data, we compute the total cost $J(w,b)$ across all $m$ examples:
$$J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left( f_{wb}(x^{(i)}) - y^{(i)} \right)^2$$

### 3. Gradient Descent Optimization
To minimize the cost function $J(w,b)$ and find the optimal parameters $w$ and $b$, the gradient descent algorithm iteratively updates the values simultaneously:
$$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$
$$w := w - \alpha \frac{\partial J(w,b)}{\partial w}$$

Where $\alpha$ represents the learning rate, and the partial derivatives (gradients) are calculated as follows:
$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{wb}(x^{(i)}) - y^{(i)} \right)$$
$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{wb}(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

## 📈 Model Performance & Training Results
* **Hyperparameters:**
    * Learning Rate ($\alpha$): `0.01`
    * Iterations: `1500`
    * Initial parameters: $w = 0.0$, $b = 0.0$
* **Convergence Log:**
    * Iteration 0: Cost = `6.74`
    * Iteration 300: Cost = `4.96`
    * Iteration 900: Cost = `4.53`
    * Iteration 1500: Cost = `4.49`
* **Optimized Parameters Found:**
    * Weight Coefficient ($w$): `1.166362`
    * Bias Parameter ($b$): `-3.630291`

## 🔮 Sample Predictions
Using the trained parameters, the franchise can forecast profitability for candidate expansion targets:
1.  **For a city with a population of 35,000 people** ($x = 3.5$):
    * Predicted Profit: **\$4,519.77 / month**
2.  **For a city with a population of 70,000 people** ($x = 7.0$):
    * Predicted Profit: **\$45,342.45 / month**
