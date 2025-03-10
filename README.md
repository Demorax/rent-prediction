# rent-prediction
These models are designed to predict rental prices based on user-defined parameters.
- CNN (Convolutional Neural Network) → MAE: 36.37
- MLP (Multi-Layer Perceptron) → MAE: 77.48
-  XGBoost (initially tested, but not included) → MAE: 108

Although XGBoost was initially considered, it did not perform as well as the deep learning models and is not included in this repository.

The lower Mean Absolute Error (MAE) for CNN suggests that it captured spatial or complex relationships better than MLP and XGBoost.

### DATA
This dataset was collected from the top four domestic real estate websites. It is organized into four primary categories: apartments, houses, land, 
and commercial properties. Each of these categories is further divided into three subcategories: auction, rent, and sale.

### Dataset Description
Apartments: Approximately 51,000 entries with 14 features. 

- id: A unique identifier for each apartment listing.
- building_type: The type of building, such as brick, panel, or other construction types.
- city: Encoded representation of the city where the apartment is located.
- condition: The condition of the apartment (e.g., new, good, or under construction).
- estate_type: Indicates the type of property (all entries pertain to apartments in this dataset).
- floor_space: The total floor area of the apartment, measured in square meters.
- land_space: Land area associated with the apartment (often 0 for apartments).
- price: The monthly rental price of the apartment, in the local currency.
- region: Encoded representation of the geographical region.
- sale_type: Indicates whether the property is for rent, sale or auction
- source: The platform or website from which the data was sourced.
- disposition: The layout or number of rooms in the apartment (e.g., 1+kk, 2+1, etc.).
- equipment: Level of furnishing in the apartment (e.g., undefined, unfurnished, partially furnished, furnished).
- penb: Energy performance certificate rating, categorized from A (best) to G (worst).

### Goals and Objectives
- The primary objective is to predict the rental prices of apartments based on the available features. The focus is 
on analyzing the influence of various factors, such as:
- Location (city and region)
- Apartment size (floor_space)
- Building condition (condition)
- Furnishing level (equipment)
- Energy efficiency (penb)
- And other relevant features.

### Hypothesis
- Rental prices are significantly influenced by the location, size, and condition of the apartment. Additional factors s
- uch as the furnishing level and energy efficiency also play an essential role in determining the rental price.

### ⚠️ Disclaimer
The dataset used in this project was collected from multiple domestic sources through web scraping. Due to uncertainty 
regarding legal implications and potential copyright or data privacy concerns, I have chosen not to publish the data online.

For research and experimentation, users should collect their own data or use publicly available datasets that comply with relevant legal and ethical guidelines.