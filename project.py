import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = "https://www.espn.in/football/table/_/league/esp.1"

try:
    tables = pd.read_html(url)
    
    table_0 = tables[0]
    
    table_0[["Rank","Teams"]] = table_0["2024-2025"].str.extract(r"(\d+)([A-Z]+.+)")
    table_0 = table_0[["Rank","Teams"]]
    #Cleaning the Teams column
    table_0["Teams"] = table_0["Teams"].str.replace(r"^[A-Z]{3}", "", regex=True)
    
    table_1 = tables[1]
    print(table_1.columns)
    table_1.columns = ["GP", "W", "D", "L", "F", "A", "GD", "P"]
    
    laliga_table = pd.concat([table_0.reset_index(drop=True),table_1.reset_index(drop=True)],axis=1)
    
    laliga_table.to_csv("laliga_table.csv",index=False)
    print("Table saved successfully")
except Exception as e:
    print("Error while downloading table: ", e)

#Loading the dataset
data = pd.read_csv("laliga_table.csv")

#Exploring the data
print(data.head())
print(data.info())
print(data.describe())

# Convert numerical columns to appropriate datatypes
numeric_cols = ["GP", "W", "D", "L", "F", "A", "GD", "P"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)

#Data Clening
data.drop_duplicates(inplace=True)

#Handle missing values
data.fillna(0, inplace=True)

#Data visualization

#Bar chart for Total Points
plt.figure(figsize=(12, 6))
sns.barplot(x="Teams", y="P", data=data, palette="viridis")
plt.title("Points by Team", fontsize=16)
plt.xticks(rotation=90)
plt.ylabel("Points")
plt.xlabel("Teams")
plt.tight_layout()
plt.show()

#Scatter Plot for Goals Scored vs Goals Conceded
plt.figure(figsize=(8, 6))
sns.scatterplot(x="F", y="A", data=data, hue="Rank", palette="cool", size="P", sizes=(50, 300))
plt.title("Goals Scored vs Goals Conceded", fontsize=16)
plt.xlabel("Goals Scored (F)")
plt.ylabel("Goals Conceded (A)")
plt.legend(title="Rank", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#Heatmap of Goal Difference (GD)
plt.figure(figsize=(10, 6))
gd_matrix = data.pivot_table(values="GD", index="Teams", columns="Rank")
sns.heatmap(gd_matrix, cmap="coolwarm", annot=True, fmt=".0f", cbar_kws={"label": "Goal Difference"})
plt.title("Goal Difference by Teams and Rank", fontsize=16)
plt.ylabel("Teams")
plt.xlabel("Rank")
plt.tight_layout()
plt.show()

#Line Plot for Win, Draw, and Loss Distribution
plt.figure(figsize=(12, 6))
data_melted = data.melt(id_vars=["Teams"], value_vars=["W", "D", "L"], var_name="Result", value_name="Count")
sns.lineplot(data=data_melted, x="Teams", y="Count", hue="Result", marker="o", palette="Set2")
plt.title("Win, Draw, Loss Distribution by Team", fontsize=16)
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.xlabel("Teams")
plt.tight_layout()
plt.show()

#Splitting the data into X and y
X = data[["W", "D", "L", "F", "A", "GD"]]
y = data["P"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ", mse)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")
plt.title("Actual vs Predicted Points")
plt.show()

#Function to predict points after 38 games for a specific row
def predict_points_for_row(row, GP=38):
    #Calculate average stats based on current games played
    avg_win_ratio = row["W"] / row["GP"] if row["GP"] != 0 else 0
    avg_draw_ratio = row["D"] / row["GP"] if row["GP"] != 0 else 0
    avg_goals_scored = row["F"] / row["GP"] if row["GP"] != 0 else 0
    avg_goals_conceded = row["A"] / row["GP"] if row["GP"] != 0 else 0

    #Scale these values to 38 games
    W = int(avg_win_ratio * GP)
    D = int(avg_draw_ratio * GP)
    L = GP - (W + D)
    F = int(avg_goals_scored * GP)
    A = int(avg_goals_conceded * GP)
    GD = F - A

    #Create input data for prediction
    input_data = pd.DataFrame([[W, D, L, F, A, GD]], columns=["W", "D", "L", "F", "A", "GD"])
    predicted_points = model.predict(input_data)[0]
    return predicted_points

#Apply the function row-wise
data["Predicted_Points_38"] = data.apply(predict_points_for_row, axis=1)

#Dropping duplicates
data = data.drop_duplicates(subset=["Teams"], keep="first")

#Display the teams and their predicted points
data["Rank"] = range(1, len(data)+1)
print(data[["Rank", "Teams", "Predicted_Points_38"]].sort_values(by="Rank").to_string(index=False))

winner = data.iloc[0]["Teams"]
print("The winner of the Laliga table is: ", winner)

relegation_teams = data.loc[data["Rank"] > 17, "Teams"].values
print("The teams relegated from the Laliga are:")
for team in relegation_teams:
    print("-",team)