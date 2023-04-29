# CS2410 NBA PREDICTION
A predictive model that lists the probability of each NBA team being the next NBA champion

### TECHNOLOGIES USED
* Sportsipy
* Beautiful Soup
* Requests
* Sklearn metrics (AUC-ROC, F1-Score, Accuracy score)
* Imblearn BalancedRandomForestClassifier

### HOW DOES IT WORK?
1. First, data about every team is pulled from [Basketball Reference](https://www.basketball-reference.com/) using sportsipy
    * This data includes overall team data of every year, such as their field goal percentage, blocks, etc.
    * Once every team's data is pulled through sportsipy, we use beautiful soup to scrape through the website again to add on the champion for every year, just because that data isn't available within sportsipy
    * When all the data is correctly pulled and merged together, we save it as a csv for easier access later down the line
2. Once we pulled the data we need, it starts getting cleaned. 
	* Remove excess data 
	* Champion column values gets converted to 1's and 0's (1 if the team is that year's champion, 0 if they aren't)
	* Drop the features that make the data noisy and decrease accuracy
		* Found the noisy data through BalanceRandomForestClassifier's "feature importances" property
		* This allows us to weigh each feature based on how much it effects the target, meaning we can easily figure out which features are noisy and which aren't
3. After the data is cleaned the model gets created
	* Data is split into test data and training data
	* Since it's unbalanced, it goes through the BalancedRandomForestClassifier, which acts both as a data balancer and as a model we can use to predict behavior
4. Evaluate the model using SKLearn's metrics
5. Create our list of predictions by passing the current year's data through the model

### HOW DO I USE IT?
1. Create a virtual enviornment using python's venv package
	* Create a directory to use as a virtual enviornment. I'm calling mine "venv"
	```mkdir venv```
	* Go into that directory and create a virtual enviornment inside of it
	```python
	cd venv  
	python3 -m venv venv
	python3 -m venv venv
	```


