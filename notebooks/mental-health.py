#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch, Hyperband


# In[76]:


# Hyperparameter
from tensorflow import keras
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Dropout


# In[44]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (20, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)


# In[46]:


df = pd.read_csv("../dataset/survey.csv")
df.head()


# In[47]:


df.info()


# In[48]:


# Clean all null values.
df.state.fillna("", inplace=True)
df.work_interfere.fillna(df.work_interfere.mode()[0], inplace=True)
df.self_employed.fillna(df.self_employed.mode()[0], inplace=True)
df.comments.fillna("None", inplace=True)
print(df.isnull().sum())


# In[49]:


# Combine country and state
df["Country"].where(
    df["state"] == "", other=df["Country"] + ", " + df["state"], inplace=True
)
df.drop("state", axis=1, inplace=True)
df.head()


# In[50]:


df.Gender.value_counts()


# In[51]:


# Clean gender's and make them either Male, Female, Non-binary

df["Gender"].replace(
    [
        "A little about you",
        "Agender",
        "All",
        "Androgyne",
        "Enby",
        "non-binary",
        "Nah",
        "something kinda male?",
        "p",
        "ostensibly male, unsure what that really means",
        "Genderqueer",
        "queer/she/they",
        "Neuter",
        "Trans woman",
        "Trans-female",
        "queer",
        "fluid",
        "fluid",
        "male leaning androgynous",
        "Female (trans)",
        "Guy (-ish) ^_^",
    ],
    "Nonbinary",
    inplace=True,
)


df["Gender"].replace(
    [
        "Cis Female",
        "F",
        "Femake",
        "Female ",
        "Female (cis)",
        "Woman",
        "femail",
        "female",
        "woman",
        "cis-female/femme",
        "f",
    ],
    "Female",
    inplace=True,
)

df["Gender"].replace(
    [
        "Cis Male",
        "Cis Man",
        "M",
        "Mail",
        "Make",
        "Mal",
        "Male ",
        "Male (CIS)",
        "Male-ish",
        "Man",
        "m",
        "cis male",
        "maile",
        "male",
        "msle",
        "Malr",
    ],
    "Male",
    inplace=True,
)
df.Gender.value_counts()


# In[52]:


# Not normalized data
df.Age.describe()


# In[53]:


def filterByAge(age):
    indeces_list = df.index[df["Age"] == age].tolist()
    return df.loc[indeces_list]


filterByAge(5)


# In[54]:


# Remove any age outliers above the age 100.
df["Age"] = df["Age"].abs()
assert len(df[df["Age"] < 0]) == 0
indeces_list = df.index[df["Age"] > 100].tolist()
df = df.drop(indeces_list)
df.Age.describe()


# In[55]:


bins = [0, 18, 30, 40, 50, 60, 70, np.inf]
labels = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
df["agerange"] = pd.cut(df.Age, bins, labels=labels, include_lowest=True)
df.agerange.value_counts().plot(kind="bar")
plt.xlabel("Age groups")
plt.ylabel("Number of people")
plt.title("Age group frequency")
plt.show()


# In[56]:


# Only United States
df_only_us = df.where(df["Country"].str.contains("United States")).dropna(axis=0)
df_only_us["Country"] = df["Country"].str.replace("United States, ", "")
df_only_us.Country.value_counts()[:10].plot(kind="bar")
plt.xticks(rotation=360)
plt.xlabel("States")
plt.ylabel("Number of US States")
plt.title("Top 10 US States frequency")
plt.show()


# In[57]:


# All Countires
df_global = df.copy()
df_global["Country"] = df.Country.str.replace(r"(^.*United States.*$)", "United States")
df_global.Country.value_counts()[:10].plot(kind="bar")
plt.xlabel("Countries")
plt.ylabel("Number of countries")
plt.title("Top 10 US Countries")
plt.show()


# In[58]:


df.drop("Timestamp", axis=1, inplace=True)
df.drop("agerange", axis=1, inplace=True)
df.drop("comments", axis=1, inplace=True)


# In[59]:


for column in list(df.columns.values):
    print(column, list(df[column].unique()))


# In[60]:


# One hot encoding
hot_encoding_features = [
    "self_employed",
    "family_history",
    "work_interfere",
    "remote_work",
    "tech_company",
    "benefits",
    "care_options",
    "wellness_program",
    "seek_help",
    "anonymity",
    "leave",
    "mental_health_consequence",
    "phys_health_consequence",
    "coworkers",
    "supervisor",
    "mental_vs_physical",
    "mental_health_interview",
    "phys_health_interview",
    "obs_consequence",
    "Gender",
]
for feature in hot_encoding_features:
    encoder = OneHotEncoder(handle_unknown="error")
    data_encoded = encoder.fit_transform(df[[feature]])
    for index, unique_feature_data in enumerate(df[feature].unique()):
        unique_feature_data = unique_feature_data.lower().replace(" ", "_")
        col_name = feature + "_" + unique_feature_data
        data_encoded_array = data_encoded.toarray()
        df[col_name] = data_encoded_array[:, index]
    df.drop(feature, axis=1, inplace=True)
df.head()


# In[61]:


no_emp_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
df.no_employees.value_counts().loc[no_emp_order].plot(kind="bar")
plt.xlabel("No. of employees")
plt.ylabel("Number of people")
plt.title("Employee frequency")
plt.show()


# In[62]:


# Ordinal encoding.
ordinal_features = ["Country", "no_employees"]
ordinal_enconder = OrdinalEncoder()
for features in ordinal_features:
    OrdFeaturesToNum = ordinal_enconder.fit_transform(df[[features]])
    df[features] = OrdFeaturesToNum
df.head()


# In[63]:


df.treatment.replace(("Yes", "No"), (1, 0), inplace=True)
df.head()


# In[64]:


corr_matrix = df.corr(method="spearman")
plt.figure(figsize=(25, 10))
sns.heatmap(corr_matrix, linewidths=0.5)


# In[65]:


# Split data for training
features = df.drop("treatment", axis=1)
labels = df["treatment"]

train_ratio = 0.80
validation_ratio = 0.15
test_ratio = 0.05

x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=1 - train_ratio, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=42,
)


# In[66]:


# Stochastic Gradient Descent (SGD) classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)


# In[67]:


cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")


# In[68]:



input_features = x_train.shape[1]
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(input_features,)),
        keras.layers.Dense(400, activation="relu"),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()


# In[69]:


# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))


# In[82]:



class MentalHealthTunerHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(Flatten(input_shape=(self.input_shape,)))
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout", min_value=0, max_value=0.9, default=0.1, step=0.01
                )
            )
        )
        model.add(
            Dense(
                hp.Int("units", min_value=32, max_value=1024, step=32, default=128),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu",
                ),
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout", min_value=0, max_value=0.9, default=0.1, step=0.01
                )
            )
        )
        model.add(
            Dense(
                hp.Int("units", min_value=32, max_value=1024, step=32, default=128),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu",
                ),
            )
        )
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-1,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            metrics=["accuracy"],
        )
        return model
print("done")


# In[88]:


model_tuner = MentalHealthTunerHyperModel(input_features)


def activateTuner(space_type):
    tuner = None
    epochs = 40
    MAX_TRIALS = 20
    EXEC_TRIALS = 2
    if space_type == "random":
        # Random Space Tuner
        tuner = RandomSearch(
            model_tuner,
            objective="val_accuracy",
            max_trials=MAX_TRIALS,
            executions_per_trial=EXEC_TRIALS,
            project_name="../hyperparameters",
        )
    elif space_type == "hyper":
        tuner = Hyperband(
            model_tuner,
            objective="val_accuracy",
            max_epochs=epochs,
            executions_per_trial=EXEC_TRIALS,
            project_name="../hyperparameters_hyperband",
        )
    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
    )
    # Get the optimal hyperparameters

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Save the best model.
    best_model.save(f"../models/tuned_mental_health_{space_type}.h5")
    print(best_model.summary())
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(
        f"""
    The random hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """
    )
    return tuner.results_summary()


activateTuner("random")


# In[94]:


activateTuner("hyper")


# In[90]:


random_search_best_model = keras.models.load_model('../models/tuned_mental_health_random.h5')
random_search_best_model.summary()


# In[91]:


loss, accuracy = random_search_best_model.evaluate(x_test,y_test)
print("Random Search Model Accuracy on Test set:", accuracy * 100, "%")


# In[92]:


hyperband_best_model = keras.models.load_model('../models/tuned_mental_health_hyper.h5')


# In[93]:


loss, accuracy = hyperband_best_model.evaluate(x_test,y_test)
print("Hyperband Search Model Accuracy on Test set:", accuracy * 100, "%")

