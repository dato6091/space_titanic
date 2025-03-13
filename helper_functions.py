import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_normalized_percentages(
    df: pd.DataFrame,
    feature1: list,
    feature2: str,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = (10, 6),
    label_format: str = ".0f",
) -> None:
    """
    Plots the normalized percentages of categorical features in `feature1`
    with respect to `feature2` as bar charts.

    The function calculates the percentage distribution of `feature2`
    within each category of `feature1`, then visualizes it using Seaborn's barplot.
    It also annotates the bars with percentage values.

    Args:
        - df (pd.DataFrame): The input dataframe containing the data.
        - feature1 (list): A list of categorical feature names to be plotted.
        - feature2 (str): The categorical feature to be used for grouping in the bar charts.
        - nrows (int, optional): Number of subplot rows. Defaults to 1.
        - ncols (int, optional): Number of subplot columns. Defaults to 1.
        - figsize (tuple, optional): Figure size for the subplots. Defaults to (10,6).
        - label_format (str, optional): Allows to modify the decimal places in the labels on top of the bars.

    Returns:
        None: Displays the bar plots and does not return anything.

    """
    _, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure ax is always iterable
    ax = np.ravel(ax)

    columns = feature1

    for i, col in enumerate(columns):
        # Compute normalized percentages
        ct = pd.crosstab(df[col], df[feature2], normalize="index")
        ct_long = ct.reset_index().melt(
            id_vars=col, var_name=feature2, value_name="Percentage"
        )
        ct_long["Percentage"] *= 100

        barchart = sns.barplot(
            data=ct_long, x=col, y="Percentage", hue=feature2, ax=ax[i]
        )

        # Add percentages on bars
        for p in barchart.patches:
            if p.get_height() > 0:
                barchart.annotate(
                    text=f"{p.get_height():{label_format}}%",
                    xy=(p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                    ha="center",
                    fontsize=10,
                )
        ax[i].set_title(f"{col} Normalized Percentage by {feature2}")
        ax[i].set_xlabel(f"{col}")
        ax[i].set_ylabel("Proportion")
        ax[i].tick_params(axis="y", left=False, labelleft=False)

    plt.tight_layout()
    plt.show()
    return None




def mode_by_category(df: pd.DataFrame, cat_feature: str, target: str) -> dict:
    """
    Function that calculates the mode of the target feature for each categorical value of 'cat_feature' to use them for imputation

    Args:
        - df (pd.DataFrame): Input data frame containing original data.
        - cat_feature (str): Categorical feature to group by.
        - target (str): target feature to calculate the mode for each category of 'cat_feature'.

    Returns:
        - dict: A dictionary containing the values of each category of 'cat_feature' as keys and the mode of the target feature as values.
    """
    return df.groupby(cat_feature)[target].apply(lambda x: x.mode().iloc[0]).to_dict()



def impute_homeplanet_nulls(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Imputes missing values in the 'HomePlanet' column using a heuristic approach.
    It first creates a dummy column flagging null values in 'HomePlanet'.
    Then, it iterates over the missing values and imputes them based on the following heuristics:
        1. If 'Group' is available and it is a key in the 'homeplanet_by_group_dict', use that value.
        2. Else, if 'CabinDeck' is available and it is a key in the 'mode_by_cabindeck', use that value.
        3. Else, if 'Destination' is available and it is a key in the 'mode_by_destination', use that value.
        4. Finally, if all else fails, use the global mode 'mode_value'.

    Args:
        - df (pd.DataFrame): The input DataFrame containing missing values for 'HomePlanet'.

    Returns:
        - pd.DataFrame: The DataFrame with missing 'HomePlanet' values imputed.
    """

    df_copy = df.copy()
    df_copy['HomePlanetMissing'] = df_copy['HomePlanet'].isna().astype(int)

    # Get the homeplanet for each group
    temp_df = df_copy[['Group', 'HomePlanet']].dropna().drop_duplicates()
    homeplanet_by_group_dict = {
        row['Group']: row['HomePlanet'] for _, row in temp_df.iterrows()
    }

    # Get homeplanet modes for each cabin deck, destination and overall mode
    homeplanet_mode_by_cabindeck = mode_by_category(df, 'CabinDeck', 'HomePlanet')
    homeplanet_mode_by_destination = mode_by_category(df, 'Destination', 'HomePlanet')
    homeplanet_mode = df['HomePlanet'].mode().iloc[0]

    # Iterate over rows with missing 'HomePlanet'
    for idx, row in df_copy[df_copy['HomePlanet'].isna()].iterrows():
        # 1. Check if 'Group' is available and find the group's 'HomePlanet'
        if pd.notna(row['Group']) and row['Group'] in homeplanet_by_group_dict and homeplanet_by_group_dict[row['Group']] is not None:
            df_copy.at[idx, 'HomePlanet'] = homeplanet_by_group_dict[row['Group']]
            continue

        # 2. If 'Group' is not available or doesn't provide a value, check 'CabinDeck' mode
        if pd.notna(row['CabinDeck']) and row['CabinDeck'] in homeplanet_mode_by_cabindeck and homeplanet_mode_by_cabindeck[row['CabinDeck']] is not None:
            df_copy.at[idx, 'HomePlanet'] = homeplanet_mode_by_cabindeck[row['CabinDeck']]
            continue

        # 3. If 'CabinDeck' is not available, check 'Destination' mode mode_by_destination
        if pd.notna(row['Destination']) and row['Destination'] in homeplanet_mode_by_destination and homeplanet_mode_by_destination[row['Destination']] is not None:
            df_copy.at[idx, 'HomePlanet'] = homeplanet_mode_by_destination[row['Destination']]
            continue

        # 4. If all else fails, use the global mode
        df_copy.at[idx, 'HomePlanet'] = homeplanet_mode

    return df_copy



def impute_cryosleep_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'CryoSleep' column using a heuristic approach.

    Steps:
    1. Create a dummy column 'CryoSleepMissing' indicating missing values.
    2. If any of the columns ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
       has a value greater than 0, set 'CryoSleep' to False.
    3. If all these columns are missing or 0, use the mode of 'CryoSleep' for the corresponding 'CabinDeck'.
    4. If 'CabinDeck' is also missing, use the overall mode of 'CryoSleep'.

    Args:
        - df (pd.DataFrame): The input DataFrame containing missing values for 'CryoSleep'

    Returns:
        - pd.DataFrame: A DataFrame with missing 'CryoSleep' values imputed.
    """

    df_copy = df.copy()

    # Create a dummy column
    df_copy["CryoSleepMissing"] = df_copy["CryoSleep"].isna().astype(int)

    # Get the mode by deck and overall mode
    cryosleep_mode_by_deck = mode_by_category(df, "CabinDeck", "CryoSleep")
    cryosleep_mode = df["CryoSleep"].mode().iloc[0]

    # Impute missing values with "False" whenever the spending on services is greater than 0
    services_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df_copy.loc[
        df_copy["CryoSleep"].isna() & (df_copy[services_columns].gt(0).any(axis=1)),
        "CryoSleep",
    ] = False

    for idx, row in df_copy[df_copy["CryoSleep"].isna()].iterrows():
        if (
            pd.notna(row["CabinDeck"])
            and row["CabinDeck"] in cryosleep_mode_by_deck
            and pd.notna(cryosleep_mode_by_deck[row["CabinDeck"]])
        ):
            df_copy.at[idx, "CryoSleep"] = cryosleep_mode_by_deck[row["CabinDeck"]]
        else:
            df_copy.at[idx, "CryoSleep"] = cryosleep_mode

    return df_copy




def impute_cabin_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'CabinDeck', 'CabinNumber', and 'CabinSide' columns using a heuristic approach.

    Steps:
    1. Create a dummy column 'CabinMissing' (integer) indicating if the cabin information was originally missing.
    2. If the passenger belongs to a group (GroupSize > 1), use the information from other passengers in the group to impute nulls.
    3. If no group information is available or helpful, impute 'CabinDeck' using the most common value for the corresponding 'HomePlanet'.
    4. Impute 'CabinNumber' as -1 for missing values.
    5. Impute 'CabinSide' randomly as 'P' or 'S' with equal probability.

    Args:
        - df (pd.DataFrame): The input DataFrame containing 'CabinDeck', 'CabinNumber', 'CabinSide', 'HomePlanet', and 'GroupSize'.

    Returns:
        - pd.DataFrame: A DataFrame with missing cabin values imputed.
    """
    df_copy = df.copy()

    # Create a dummy column indicating missing cabin information
    df_copy["CabinMissing"] = df_copy["CabinDeck"].isna().astype(int)

    # Create a dictionary with the groups as keys and the cabin as values
    group_cabin_dict = {
        row['Group']: {
            'CabinDeck': row['CabinDeck'],
            'CabinNumber': row['CabinNumber'],
            'CabinSide': row['CabinSide']
            }
        for _, row in df_copy.dropna(subset=['CabinDeck']).iterrows()
    }

    # Impute missing cabin values based on other passengers in the same group
    for idx, row in df_copy[df_copy["CabinDeck"].isna()].iterrows():
        if row['Group'] in group_cabin_dict:
            df_copy.at[idx, "CabinDeck"] = group_cabin_dict[row['Group']]['CabinDeck']
            df_copy.at[idx, "CabinNumber"] = group_cabin_dict[row['Group']]['CabinNumber']
            df_copy.at[idx, "CabinSide"] = group_cabin_dict[row['Group']]['CabinSide']

    # Compute mode of 'CabinDeck' for each 'HomePlanet'
    deck_mode_by_homeplanet = df_copy.groupby("HomePlanet")["CabinDeck"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    # Apply mode by 'HomePlanet' where applicable
    for idx, row in df_copy[df_copy["CabinDeck"].isna()].iterrows():
        if (
            pd.notna(row["HomePlanet"])
            and row["HomePlanet"] in deck_mode_by_homeplanet
            and pd.notna(deck_mode_by_homeplanet[row["HomePlanet"]])
        ):
            df_copy.at[idx, "CabinDeck"] = deck_mode_by_homeplanet[row["HomePlanet"]]

    # Step 5: Impute missing 'CabinNumber' with -1
    df_copy["CabinNumber"].fillna(-1, inplace=True)

    # Step 6: Impute missing 'CabinSide' randomly as 'P' or 'S'
    df_copy.loc[df_copy["CabinSide"].isna(), "CabinSide"] = np.random.choice(
        ["P", "S"], size=df_copy["CabinSide"].isna().sum()
    )

    return df_copy




def impute_destination_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'Destination' column using a hierarchical approach.

    Steps:
    1. If the passenger belongs to a group, use the first known 'Destination' of that group.
    2. If no group information is available or doesn't help, use the mode of 'Destination'
       for the corresponding 'HomePlanet'.
    3. If 'HomePlanet' is also missing, use the overall mode of 'Destination'.

    Args:
        - df (pd.DataFrame): The input DataFrame containing 'Destination', 'Group', and 'HomePlanet'.

    Returns:
        pd.DataFrame: A DataFrame with missing 'Destination' values imputed.
    """

    df_copy = df.copy()
    df_copy["DestinationMissing"] = df_copy["Destination"].isna().astype(int)

    # Get the mode by homeplanet and overall mode
    destination_mode_by_homeplanet = mode_by_category(df, 'HomePlanet', 'Destination')
    destination_mode = df['Destination'].mode().iloc[0]

    # Create a dictionary mapping groups to the first available Destination in the group
    group_destination_dict = (
        df_copy.dropna(subset=['Destination'])
        .groupby('Group')['Destination']
        .first()
        .to_dict()
    )

    # Iterate over rows with missing 'Destination'
    for idx, row in df_copy[df_copy['Destination'].isna()].iterrows():
        # 1. Check if 'Group' is available and has a known 'Destination'
        if pd.notna(row['Group']) and row['Group'] in group_destination_dict:
            df_copy.at[idx, 'Destination'] = group_destination_dict[row['Group']]
            continue  # Move to next row

        # 2. If 'Group' is not available or doesn't help, check 'HomePlanet' mode
        if (
            pd.notna(row['HomePlanet'])
            and row['HomePlanet'] in destination_mode_by_homeplanet
            and pd.notna(destination_mode_by_homeplanet[row['HomePlanet']])
        ):
            df_copy.at[idx, 'Destination'] = destination_mode_by_homeplanet[row['HomePlanet']]
            continue

        # 3. If all else fails, use the global mode
        df_copy.at[idx, 'Destination'] = destination_mode

    return df_copy




def impute_vip_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values for the VIP column and creates a new column flagging missing values as 1.
    The imputation is done by filling missing values with the mode (which in this case is 0).

    Args:
        - df (pd.DataFrame): Input data frame containing the VIP column with missing values.

    Returns:
        - pd.DataFrame: A new data frame with missing values imputed and the new column.
    """
    df_copy = df.copy()
    df_copy['VIPMissing'] = df_copy['VIP'].isna().astype(int)
    vip_mode = df_copy['VIP'].mode()[0]
    df_copy['VIP'] = df_copy['VIP'].fillna(vip_mode)
    return df_copy




def impute_age_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values for the Age column with the median value from the input data frame.
    Creates a new column 'AgeMissing' to flag missing values.
    Creates a new column 'AgeGroup' to bin ages into groups of 10 years (labeled 0-7).

    Args:
        - input_df (pd.DataFrame): The input data frame with the missing values of 'Age'.

    Returns:
        - pd.DataFrame: A new data frame with missing values imputed and new columns added.
    """

    df_copy = df.copy()
    df_copy['AgeMissing'] = df_copy['Age'].isna().astype(int)

    median_age = df_copy['Age'].median()

    # Impute missing values with the median
    df_copy['Age'] = df_copy['Age'].fillna(median_age)
    df_copy['AgeGroup'] = df_copy['Age'].apply(lambda x: str(int(x // 10)))
    age_categories = [str(i) for i in range(0, 8)]  # Age groups 0-7

    # Convert to an ordered categorical variable
    df_copy['AgeGroup'] = pd.Categorical(df_copy['AgeGroup'], categories=age_categories, ordered=True)
    df_copy.drop(columns='Age', inplace=True)

    return df_copy




def impute_amenities_nulls(df: pd.DataFrame, amenity: str) -> pd.DataFrame:
    """
    Imputes missing values for the amenity passed as an argument.
    First it checks if the passenger was on CryoSleep, if so, it imputes a 0. Else, it will use the median

    Args:
        - df (pd.DataFrame): Input data frame containing the amenity column with missing values.
        - amenity (str): The name of the amenity to impute.

    Returns:
        - pd.DataFrame: A new data frame with missing values imputed.
    """

    df_copy = df.copy()

    median_value = df_copy[amenity].median()
    for idx, row in df_copy[df_copy[amenity].isna()].iterrows():
        if row['CryoSleep']:
            df_copy.at[idx, amenity] = 0
        else:
            df_copy.at[idx, amenity] = median_value

    return df_copy




