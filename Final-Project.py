# %%
import pandas as pd
import re
import matplotlib as plt
from datetime import datetime

# %%
olypmic_df = pd.read_csv("Datasets/Olympic_Athlete_Bio.csv", index_col="athlete_id")
# %%
# create a new dataframe that contains only the entries that have date in yyyy-mm-dd format in the born column

# # define a regular expression pattern to match 'yyyy-mm-dd'
regex_pattern = "\d{4}-\d{2}-\d{2}"

# # create a boolean mask to select rows where 'born' matches the pattern
mask = olypmic_df["born"].apply(lambda x: bool(re.match(regex_pattern, x)))

# # create a new DataFrame with only the rows where 'born' matches the pattern
clean_df = olympic_df[mask]

# %%
# create a new dataframe that contains only the entries that are older than September 1st 1994 (This is where our birth data begins)

# # convert the born column into a datetime object
clean_df["born"] = pd.to_datetime(clean_df["born"])

# # create a datetime object for September 1st, 1994
sep_1_1994 = datetime(1994, 9, 1)

# # create a boolean mask to select rows where the birth date is after September 1st, 1994
mask = clean_df["born"] > sep_1_1994

# # create a new DataFrame with only the rows where the birth date is before September 1st, 1994
clean_1994_df = clean_df[mask]


# %%
def days_since_sep1(date):
    born_date = date
    sep1_date = datetime(born_date.year, 9, 1)
    delta = born_date - sep1_date
    if int(delta.days) < 0:
        sep1_date = datetime(born_date.year - 1, 9, 1)
        delta = born_date - sep1_date
    return delta.days


# %%

clean_1994_df["days_since_sep1"] = clean_1994_df["born"].apply(days_since_sep1)
clean_1994_df["birthyear"] = clean_1994_df["born"].dt.year

clean_1994_df["yyyy-mm-dd"]
# %%

# group the entries by birth year and days since September 1st and count the number of entries in each group
counts_df = (
    clean_1994_df.groupby(["birthyear", "days_since_sep1"])["born"]
    .count()
    .reset_index()
)
# %%
births_df = pd.read_csv("Datasets/US_births_1994-2014.csv")
births_df["births_by_year"] = births_df.groupby(["year"])["births"].transform("sum")
births_df["density"] = births_df["births"] / births_df["births_by_year"]
# %%
counts_df
# %%
births_df
# %%
