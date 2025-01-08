# Abstract
One of the main challenges faced in urban areas is enhancing the mobility of people
to make it as efficient, fast, and environmentally sustainable as possible. A key
component in achieving this goal is promoting active mobility which can be defined
as transportation methods that are low-cost, zero-emissions, and also offer physical
benefits by supporting a more active lifestyle. One of the most widely diffused active
ways of moving nowadays, besides walking, is cycling. In order to promote the use
of the latter, modern, safe, and comfortable bikes should be studied.
One of the possible features that such a modern lightweight vehicle should have
is terrain recognition, defined as the ability to detect the type of terrain the vehicle
is traversing based on sensor data. In this study this functionality has been implemented on a sensor-equipped e-bike prototype, enabling the distinction between
three terrains: asphalt, gravel, and cobblestone. The process to achieve this objective began with the collection of sensor measurements on the three terrains in
order to create a labeled dataset. Once collected, the data was cleaned and then
analyzed in both the time and frequency domains to extract relevant features for
terrain differentiation. Finally, a machine learning algorithm capable of determining
terrains based on the measurements observed by the sensor has been implemented.
This model was trained and optimized to improve accuracy in terrain recognition,
allowing reliable real-time predictions.
The methodology presented in this study can be followed as a guideline to implement
new terrain recognition algorithms using inertial measurements or to improve this
work by adding terrains into the current model. The practical result of this work is
an algorithm able to recognize the three terrains on which the model is trained.


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

## Helper functions
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
