# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# %%
# Load titanic_full.csv
data = pd.read_csv("../data/titanic full.csv")

# %%

data["sex"].unique()
encoding = OneHotEncoder()

# %%
# One hot encode the passenger gender field
gender = encoding.fit_transform(data[["sex"]])
gender_dataframe = pd.DataFrame(gender.toarray(), columns=encoding.categories_[0])
data = pd.concat([data, gender_dataframe], axis=1)


# %%
# One hot encode the passenger cabin
data["cabin"] = data["cabin"].fillna("unknown")
cabin_encoding = encoding.fit_transform(data[["cabin"]])
cabin_dataframe = pd.DataFrame(
    cabin_encoding.toarray(), columns=encoding.categories_[0]
)
data = pd.concat([data, cabin_dataframe], axis=1)

# %%
# Prepare a dataframe which is appropriate to predict whether a passenger survived based on age, sex, cabin and pclass
data = data[
    ["age", "female", "male"] + list(cabin_dataframe.columns) + ["pclass", "survived"]
].dropna()
data

# %%
# Load ifi_2023.csv
data2 = pd.read_csv("../data/wifi_2023.csv", encoding="cp1252")
data2

# %%
# Create one hot encoded day of week columns based off the firstSeen field.
firstSeen = pd.to_datetime(data2["FirstSeen"])
weekofday = firstSeen.dt.day_name()
e_weekofday = encoding.fit_transform(weekofday.values.reshape(-1, 1))
dowdataframe = pd.DataFrame(e_weekofday.toarray(), columns=encoding.categories_[0])
data2 = pd.concat([data2, dowdataframe], axis=1)
data2

# %%
# Create a cross feature which represents lat, lon and altitude.  Make this feature categorical - it should have 10 possible values.
data2["lat_lon_alt"] = pd.cut(
    data2["CurrentLatitude"] * data2["CurrentLongitude"] * data2["AltitudeMeters"],
    bins=10,
    labels=range(10),
)
data2
