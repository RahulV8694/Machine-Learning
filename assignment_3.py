# %%
import pandas as pd
import matplotlib.pyplot as pl


# %%
# Reading file https://clarksonmsda.org/datafiles/commute/ for 41
data = pd.read_csv("../data/commute41.csv")
print(data)

# %%
# Create a scatter plot which helps to show the relationship between sun_pct and commute_type
sun_pct = data["sun_pct"]
commute_method = data["commute_method"]
pl.xlabel("Commute type")
pl.ylabel("Sun percentage")
pl.title("Scatter plot of commute depending on the Sun percentage")
pl.scatter(commute_method, sun_pct)

# %%
# Using itterative approach to find the best sun_pct threshold overall for the entire dataset. with rate 1

thres = 20
rate = 1
acc_list = []
threshold_list = []
while thres < 80:

    def predict(row):
        if row["sun_pct"] > thres:
            return "Foot"
        else:
            return "car"

    data["predicted"] = data.apply(predict, axis=1)
    data["correct_int"] = (data["predicted"] == data["commute_method"]).astype(int)
    acc = data["correct_int"].mean()
    acc_list.append(acc)
    threshold_list.append(thres)
    thres += rate


# %%
max_accuracy = max(acc_list)
max_threshold = max(threshold_list)
print(
    "The maximum accuracy is: "
    + str(max_accuracy)
    + " at threshold: "
    + str(max_threshold)
)


# %%
# scatter plot of the accuracy per iteration of your loop.

pl.scatter(threshold_list, acc_list)
pl.xlabel("Threshold")
pl.ylabel("Accuracy")
pl.title("Scatter plot of the accuracy per iteration of your loop")
pl.show()

# %%
# Using itterative approach to find the best sun_pct threshold overall for the entire dataset. with rate 2
thres = 20
rate = 2
acc_list = []
threshold_list = []
while thres < 80:

    def predict(row):
        if row["sun_pct"] > thres:
            return "Foot"
        else:
            return "car"

    data["predicted"] = data.apply(predict, axis=1)
    data["correct_int"] = (data["predicted"] == data["commute_method"]).astype(int)
    acc = data["correct_int"].mean()
    acc_list.append(acc)
    threshold_list.append(thres)
    thres += rate
pl.scatter(threshold_list, acc_list)

# %%
# Using itterative approach to find the best sun_pct threshold overall for the entire dataset. with rate 5
thres = 20
rate = 5
acc_list = []
threshold_list = []
while thres < 80:

    def predict(row):
        if row["sun_pct"] > thres:
            return "Foot"
        else:
            return "car"

    data["predicted"] = data.apply(predict, axis=1)
    data["correct_int"] = (data["predicted"] == data["commute_method"]).astype(int)
    acc = data["correct_int"].mean()
    acc_list.append(acc)
    threshold_list.append(thres)
    thres += rate
pl.scatter(threshold_list, acc_list)
