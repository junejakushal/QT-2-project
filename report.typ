= Introduction

*Problem Statement* — This project analyses the Boston Housing dataset to model and predict median residential property values (MEDV) from neighbourhood- and property-level features such as crime rate, proportion of residential land zoned for lots, average number of rooms per dwelling, the proportion of owner-occupied units built prior to 1940, distances to employment centres, property-tax rate, and air pollution. We build and compare predictive models to estimate house prices and identify the most influential variables.

*Rationale* — Housing prices are central to decisions made by homeowners, buyers, lenders, investors, and urban planners. Prices vary with location, neighbourhood characteristics, housing quality, and local amenities; understanding which features most strongly affect value helps stakeholders make informed pricing, investment, and policy decisions. Accurate predictive models also assist lenders with risk assessment and help policymakers target interventions to stabilise or improve local housing markets.

*Application used* — Jupyter Notebook (R), R Studio, VS Code, Google Colab and Github for remote repository.

= Data Description

We have used the Boston Housing dataset originally compiled by Harrison and Rubinfeld (1978) and made widely available through the UCI Machine Learning Repository and various machine learning libraries. The dataset is accessible via R's `mlbench` module and also available on Kaggle and other data repositories. The data contains 506 observations (arranged as rows) representing different census tracts in the Boston area, and 14 variables (arranged as columns). The dataset includes 13 predictor variables (continuous and categorical) and 1 target variable (CMEDV) representing the median value of owner-occupied homes.

#table(
  columns: 3,
  stroke: 0.5pt,
  align: (left, left, left),
  table.header([*Variable*], [*Characteristic*], [*Description*]),
  [CRIM], [Continuous], [Per capita crime rate by town],
  [ZN], [Continuous], [Proportion of residential land zoned for lots over 25,000 sq.ft.],
  [INDUS], [Continuous], [Proportion of non-retail business acres per town],
  [CHAS], [Categorical - binary], [Charles River dummy variable (1 if tract bounds river; 0 otherwise)],
  [NOX], [Continuous], [Nitric oxides concentration (parts per 10 million)],
  [RM], [Continuous], [Average number of rooms per dwelling],
  [AGE], [Continuous], [Proportion of owner-occupied units built prior to 1940],
  [DIS], [Continuous], [Weighted distances to five Boston employment centres],
  [RAD], [Discrete], [Index of accessibility to radial highways (1-24)],
  [TAX], [Continuous], [Full-value property-tax rate per \$10,000],
  [PTRATIO], [Continuous], [Pupil-teacher ratio by town],
  [B], [Continuous], [$1000(B_k - 0.63)^2$ where $B_k$ is the proportion of Black residents by town],
  [LSTAT], [Continuous], [Percentage of lower status of the population],
  [CMEDV], [Continuous], [Median value of owner-occupied homes in \$1000's (target variable)],
)

The dataset also includes fields such as town name, longitude, latitude, and census tract identifier. These fields are not used in the analysis as they do not contribute to the predictive modeling of house prices, but they are useful for geospatial analysis or mapping purposes.


= Exploratory Data Analysis (EDA)

*Summary Statistics* — We begin with summary statistics for each variable, including minimum, first quartile, median, mean, third quartile, maximum, skewness, and kurtosis values. This provides an initial understanding of the data distribution, central tendency, potential outliers, and the shape of distributions.

#table(
  columns: 9,
  stroke: 0.5pt,
  align: (left, right, right, right, right, right, right, right, right),
  table.header([*Variable*], [*Min*], [*1st Qu.*], [*Median*], [*Mean*], [*3rd Qu.*], [*Max*], [*Skewness*], [*Kurtosis*]),
  [cmedv], [5.00], [17.02], [21.20], [22.53], [25.00], [50.00], [1.108], [4.490],
  [crim], [0.00632], [0.08205], [0.25651], [3.61352], [3.67708], [88.97620], [5.208], [39.753],
  [zn], [0.00], [0.00], [0.00], [11.36], [12.50], [100.00], [2.219], [6.980],
  [indus], [0.46], [5.19], [9.69], [11.14], [18.10], [27.74], [0.294], [1.767],
  [nox], [0.3850], [0.4490], [0.5380], [0.5547], [0.6240], [0.8710], [0.727], [2.924],
  [rm], [3.561], [5.886], [6.208], [6.285], [6.623], [8.780], [0.402], [4.861],
  [age], [2.90], [45.02], [77.50], [68.57], [94.08], [100.00], [-0.597], [2.030],
  [dis], [1.130], [2.100], [3.207], [3.795], [5.188], [12.127], [1.009], [3.471],
  [rad], [1.000], [4.000], [5.000], [9.549], [24.000], [24.000], [1.002], [2.129],
  [tax], [187.0], [279.0], [330.0], [408.2], [666.0], [711.0], [0.668], [1.857],
  [ptratio], [12.60], [17.40], [19.05], [18.46], [20.20], [22.00], [-0.800], [2.706],
  [b], [0.32], [375.38], [391.44], [356.67], [396.23], [396.90], [-2.882], [10.144],
  [lstat], [1.73], [6.95], [11.36], [12.65], [16.95], [37.97], [0.904], [3.477],
)

To better understand our dataset, we undertake univariate and bivariate analyses, visualising distributions and relationships between variables.

// crim
#image("assets/image-1.png")

// zn
#image("assets/image-2.png")

// indus
#image("assets/image-3.png")

// nox
#image("assets/image-4.png")


// rm
#image("assets/image-5.png")

// age
#image("assets/image-6.png")

// dis
#image("assets/image-7.png")

// tax
#image("assets/image-8.png")

// ptratio
#image("assets/image-9.png")

// B
#image("assets/image-10.png")

// lstat
#image("assets/image-11.png")

// cmedv
#image("assets/image.png")


// properties by charles river proximity
#image("assets/image-12.png")

