# %%
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime
import scipy

# %%
olympic_df = pd.read_csv("Datasets/Olympic_Athlete_Bio.csv", index_col="athlete_id")
results_df = pd.read_csv("Datasets/Olympic_Athlete_Event_Results.csv", index_col="athlete_id")
births_df = pd.read_csv("Datasets/US_births_1994-2014.csv")


# %%
def clean_data(df):
    """
    This function takes in our dataset and returns a dataframe that contains athletes whose birthdates are known and have been born since Sept 1st 1994
    """
    # # define a regular expression pattern to match 'yyyy-mm-dd'
    regex_pattern = "\d{4}-\d{2}-\d{2}"
    # # create a boolean mask to select rows where 'born' matches the pattern
    mask = df["born"].apply(lambda x: bool(re.match(regex_pattern, x)))

    # # create a new DataFrame with only the rows where 'born' matches the pattern
    clean_df = df[mask]

    # create a new dataframe that contains only the entries that are older than September 1st 1994 (This is where our birth data begins)

    # # convert the born column into a datetime object
    clean_df["born"] = pd.to_datetime(clean_df["born"])

    # # create a datetime object for September 1st, 1994
    sep_1_1994 = datetime(1994, 9, 1)

    # # create a boolean mask to select rows where the birth date is after September 1st, 1994
    mask = clean_df["born"] > sep_1_1994

    # # create a new DataFrame with only the rows where the birth date is before September 1st, 1994
    clean_1994_df = clean_df[mask]

    # drop all the columns that we don't care about and count the number of times each athlete has played each sport
    reduced_results_df = results_df.groupby(["athlete_id", "sport"]).size().reset_index(name='count')
    # select the sport for each athlete that they have played the most times:
    idx = reduced_results_df.groupby(["athlete_id"])['count'].idxmax()
    reduced_results_df = reduced_results_df.loc[idx]
    clean_1994_df = pd.merge(clean_1994_df, reduced_results_df["athelete_id", "sport"], on="athlete_id", how="inner")  

    return clean_1994_df


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
def join_athlete_and_birth_data(clean_1994_df, births_df):
    """
    This function takes in the clean_1994_df and births_df and returns a combined dataframe that contains the distributions that can then be graphed
    """
    # Compute the average number of people born per year
    births_df["births_by_year"] = births_df.groupby(["year"])["births"].transform("sum")

    # Compute the density of births
    births_df["births_density"] = births_df["births"] / births_df["births_by_year"]

    # Add a "yyyy-mm-dd" column so that the births_df can be joined to the athlete data
    births_df["yyyy-mm-dd"] = (
        births_df["year"].astype(str)
        + "-"
        + births_df["month"].astype(str).str.zfill(2)
        + "-"
        + births_df["date_of_month"].astype(str).str.zfill(2)
    )
    # Drop data from the births dataframe that is past 2006. At this point we no longer have good data for athletes
    # convert yyyy-mm-dd to a datetime object
    births_df["yyyy-mm-dd"] = pd.to_datetime(births_df["yyyy-mm-dd"])

    # create a boolean mask for the rows where year is greater than 2006
    mask = births_df["yyyy-mm-dd"].dt.year > 2006

    # use boolean indexing to drop the rows where year is greater than 2006
    births_df = births_df.loc[~mask]

    # Calculate the days_since_sep1 for each date
    births_df.loc[:, "days_since_sep1"] = births_df["yyyy-mm-dd"].apply(days_since_sep1)

    # Add a "yyyy-mm-dd" column to the athletes table
    clean_1994_df["yyyy-mm-dd"] = clean_1994_df["born"]

    # group the entries by birth date and days since September 1st and count the number of entries in each group
    counts_df = clean_1994_df.groupby(["yyyy-mm-dd"])["born"].count().reset_index()

    # Compute normalized the athlete born data based on the corresponding yearly sum
    counts_df["year"] = pd.to_datetime(counts_df["yyyy-mm-dd"].dt.year)
    yearly_sum = counts_df.groupby("year")["born"].sum()
    counts_df["born_normalized"] = counts_df["born"] / counts_df["year"].map(yearly_sum)

    # Join the counts_df and the births_df to create the
    counts_births_df = pd.merge(births_df, counts_df, on="yyyy-mm-dd", how="left")
    counts_births_df["born"] = counts_births_df["born"].fillna(0)
    counts_births_df["born_normalized"] = counts_births_df["born_normalized"].fillna(0)

    return counts_births_df


# %%
def create_distribution(counts_births_df):
    # Create an empty DataFrame to store the averages
    averages_df = pd.DataFrame(
        columns=["days_since_sep1", "born_normalized_mean", "births_density_mean"]
    )

    # Group the rows based on the value in the "days_since_sep1" column
    day_groups = counts_births_df.groupby("days_since_sep1")

    # Iterate through each group
    for day, group in day_groups:
        # Calculate the mean of the "born_normalized" and "births_density" columns
        born_normalized_mean = group["born_normalized"].mean()
        births_density_mean = group["births_density"].mean()

        # Add a new row to the "averages_df" DataFrame with the day and the calculated means
        new_row = {
            "days_since_sep1": day,
            "born_normalized_mean": born_normalized_mean,
            "births_density_mean": births_density_mean,
        }
        averages_df = pd.concat([averages_df, pd.DataFrame([new_row])])

    return averages_df


# %%
def remove_outliers(df):
    q1 = df["born_normalized_mean"].quantile(0.25)
    q3 = df["born_normalized_mean"].quantile(0.75)
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the upper and lower bounds for outliers
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # Filter the dataframe to remove extreme outliers
    df = df[
        (df["born_normalized_mean"] >= lower_bound)
        & (df["born_normalized_mean"] <= upper_bound)
    ]
    return df


def create_graphs(df):
    averages = df
    # Create a line plot with "days_since_sep1" as the x variable and "born_normalized_mean" as one line and "births_density_mean" as another line
    plt.plot(averages["days_since_sep1"], averages["born_normalized_mean"], label="Professional Athletes")
    plt.plot(averages["days_since_sep1"], averages["births_density_mean"], label="Total Population")

    # Add axis labels and a legend
    plt.xlabel("Days Since September 1")
    plt.ylabel("Percentage of Population Born")
    plt.legend()

    # Show the plot
    plt.show()

    # Group the rows into sets of 30 based on 'days_since_sep1' and calculate the mean of the remaining columns in each group
    averages_30 = averages.groupby(averages["days_since_sep1"] // 30 * 30).mean()

    # Create a line plot with "days_since_sep1_30" as the x variable and "born_normalized_mean" as one line and "births_density_mean" as another line
    plt.plot(averages_30["days_since_sep1"], averages_30["born_normalized_mean"], label="Professional Athletes")
    plt.plot(averages_30["days_since_sep1"], averages_30["births_density_mean"], label="Total Population")

    # Add axis labels and a legend
    plt.xlabel("Days Since September 1 (30 Day Intervals)")
    plt.ylabel("Percentage of Population Born")
    plt.legend()

    # Show the plot
    plt.show()


# %%
clean_df = clean_data(olympic_df)
counts_births_df = join_athlete_and_birth_data(clean_df, births_df)
averages_df = create_distribution(counts_births_df)
create_graphs(averages_df)

# %%
scipy.stats.kstest(
    counts_births_df["born_normalized"], counts_births_df["births_density"]
)
