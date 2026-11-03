# titanic-survival-prediction-python
Predicts whether a passenger survived the Titanic disaster from details such as class, sex, age, fare and family size, using a scikit-learn pipeline.

**Dataset:** [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic) - `train.csv`, 891 passengers.

## Approach

- Explored missing values (Age ~20%, Cabin ~77%, Embarked 2 rows) and visualised survival by sex, class, age, port and fare
- Checked Fare outliers with the IQR method (116 found, kept as genuine first-class fares)
- Engineered `FamilySize` and `IsAlone`; dropped Cabin, Name, Ticket, PassengerId
- Split train/test (80/20, stratified) **before** any imputation or scaling to avoid data leakage
- Built one `Pipeline` + `ColumnTransformer`: median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for categorical ones
- Compared three models with 5-fold stratified cross-validation, then tuned a Random Forest with `GridSearchCV`
- Evaluated once on the held-out test set and saved the model with `pickle`

## Results

| Model | Mean CV accuracy (train set) |
|---|---|
| Logistic Regression | 0.803 |
| Random Forest | 0.802 |
| K-Nearest Neighbors | 0.792 |

Tuned Random Forest (`max_depth=6`, `n_estimators=100`, `min_samples_leaf=1`):

- 5-fold CV accuracy: **0.830**
- Held-out test accuracy: **0.805** (179 passengers)
- Recall: 0.93 for "did not survive", 0.61 for "survived"

Most important features: **Sex**, **Fare**, **Age**, third class, family size.

## Limitations

Small dataset, so test accuracy can move a few points with a different split. The model misses many survivors. Possible improvements: extract titles (Mr/Mrs/Miss) from names, use cabin deck.

## Run it

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

Download `train.csv` from the Kaggle link above, place it next to the notebook, then open `titanic_survival.ipynb` and run all cells.

## Files

- `titanic_survival.ipynb` - full analysis and model
- `titanic_model.pkl` - saved trained pipeline
- `README.md` - this file

