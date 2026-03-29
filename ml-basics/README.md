## 💗 5 Python Libraries You Actually Need for ML

If you're starting with machine learning, you don’t need dozens of libraries.

These 5 are enough to build real projects 👇

### Core stack
- **pandas** → data manipulation  
- **numpy** → numerical operations  
- **matplotlib / seaborn** → visualization  
- **scikit-learn** → ML models  
- **xgboost** → advanced models  

---

### Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
