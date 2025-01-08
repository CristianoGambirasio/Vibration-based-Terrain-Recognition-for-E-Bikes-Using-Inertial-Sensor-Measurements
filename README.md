# Repository Description
## Scripts
1) data_analysis.m -> main model training file 
	- Data loading and visualization
	- Preprocessing
	- Dataset creation
		+ grouping data
		+ computing temporal feature identified from literature
		+ computing frequency feature extracted and studied in the script "frequency_analysis.m"
	- SVM model training
	- Feature importance analysis

2) frequency_analysis.m -> investigates the impact of specific conditions on frequency response of selected imu signals
	- Assistance level 
	- Bike velocity
	- Damper 
	- Terrains
	- Identification of frequencies that can characterize independently of other factors

3) online_simulation.m -> perform real-time prediction simulations processing sample-by-sample a test dataset
	- SVM based prediction
	- HMM based output modeling
	- Prediction stability logic

# Helper functions
1) create_features.m
	used in "online_simulation.m", recives as input a 3 wheel revolutions dataset of raw data and returns features for the trained model

2) create_rev_groups.m
	function that divides a dataset in groups of 3 wheel revolutions

3) graphs.m
	custom data visualization function

4) windowed_fft.m
	used in "frequency_analysis.m" to reduce resolution of FFTs (too defined if using the entire dataset in one computation) 

## Sensor data
1) Terrain's folders
	contain sensors data of the esperiments

2) Behaviors 
	data for different riding behaviors characterization (not in this project)

3) Test 
	data and labels for the two test experiments

## Experiments list
Excel table containing the planned experiments, the timestamp of those collected and which are missing
