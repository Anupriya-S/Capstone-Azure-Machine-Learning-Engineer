# Asteroid Classification

Asteroids are minor planets, especially of the inner Solar System. Larger asteroids have also been called planetoids. The study of asteroids is also crucial as historical events prove some of them being hazardous. Like the one that probably sweeped all the dinosaurs from the face of Earth.

For the purpose of this Capstone project, I thought of using machine learning to predict whether an asteroid could be hazardous or not.

## Dataset

### Overview
The data is about Asteroids and is provided by NEOWS(Near-Earth Object Web Service). It is a NASA's dataset and can be found on Kaggle. One can download the dataset from [this](https://www.kaggle.com/shrutimehta/nasa-asteroids-classification/download) link.

The dataset contains various information about the asteroids and labels each asteroid as hazardous(1) or non-hazardous(0). The dataset consists of 4687 data instances(rows) and 40 features(columns). Although we will not be using all those features beacause some of them are highly correlated. Some of those features are:

1. Absolute Magnitude: An asteroid’s absolute magnitude is the visual magnitude an observer would record if the asteroid were placed 1 Astronomical Unit (AU) away, and 1 AU from the Sun and at a zero phase angle.
2. Est Dia in KM(min): This feature denotes the estimated diameter of the asteroid in kilometres (KM).
3. Relative Velocity km per sec: This feature denotes the relative velocity of the asteroid in kilometre per second.
4. Jupiter Tisserand Invariant: This feature denotes the Tisserand’s parameter for the asteroid. Tisserand’s parameter (or Tisserand’s invariant) is a value calculated from several orbital elements(semi-major axis, orbital eccentricity, and inclination) of a relatively small object and a more substantial‘ perturbing body’. It is used to distinguish different kinds of orbits.
5. Eccentricity: The axis marked eccentricity is a measure of how far from circular each orbit is: the smaller the eccentricity number, the more circular the realm.

These are only a few of the features that we are going to use.

### Task
As mentioned earlier, the task is to classify the asteroids as hazardous(1) or non-hazardous(0). This makes it a two class classification problem. There are 40 features in this dataset but we will get rid of 18 of them because they are highly correlated. However, I will 21 of the remaining features for the training purpose and 'Hazardous' (0 or 1) will be the target column.

### Access
For the purpose of this project, I downloaded this dataset and saved it in the project's GitHub repository and accessing it using [this](https://raw.githubusercontent.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/main/nasa.csv) link. Following screenshot shows the same:

![dataset_access](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/dataset_access.png)

## Automated ML
For the AutoML run we use the following settings:
1. Experiment timeout is set to 30 minutes because we want our AutoML run to complete in a given timeframe.
2. Maximum Concurrent Iterations are set to 5 because the upper limit of nodes for our compute cluster is set to 6.
3. Primary metric is set to Accuracy so that we can measure the 'goodness' of our model.

For the same AutoML run we use the following configurations:
1. Set the compute_target to "cpu-cluster".
2. Task is set to classification for obvious reasons.
3. Training data is set to use our training data.
4. Label column name holds the name of our target column ("Hazardous" in our case)
5. Value of n for n cross validations is set to 5 so that our model can be properly validated.
6. Enable early stopping is set to True so that our experiment does not waste time once the performance of models starts deteriorating.
7. Featurization is set to 'auto'.
8. Debug_log has the name of the file for writing the logs.

The above values can also be seen in the following screenshot:

![automl_config](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/automl_config.png)

### Results
As a result of the AutoML run we got the following models alongwith different accuracies: (a list of all the models trained by AutoML can be found as output of one of the cells in `automl.ipynb`.

![automl_allmodels](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/automl_allmodels.png)

Out of several models trained by AutoML **Voting Ensemble** gave the highest value of accuracy, **0.9964**. The run ID, accuracy, and the parameters of the model can be seen in the following screenshot:

![automl_bestmodel](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/automl_bestmodel.png)

We even used the `RunDetails` widget to monitor the run:

![automl_RunDetails](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/automl_RunDetails.png)

### Future Work
We can try following things for better results in future:
1. As reported, there was class imbalance in the dataset. For this we can use resampling of the training data.
2. In AutoMLConfig, increase experiment_timeout_minutes so that more possibilities can be explored.
3. In AutoMLConfig, set enable_dnn to True for exploring Deep Neural Networks too.

## Hyperparameter Tuning
Since this a classification problem we are using Decision Tree Classifier. It is the most powerful and popular tool for classification and prediction. It can handle high dimensional data and has high accuracy in general. And because our dataset has a large number of features, that's why, Decision Tree Classifier is a good choice.

For defining the parameter sampler, we used the Random Sampling method. The benefit of using Random sampling over any other method is that it picks up the parameters' values randomly that saves time, and the result is almost as good as any other method.

In this case, the parameter search space consists of three hyperparameters:
1. Maximum Depth of the Decision tree (5, 10, 15, 20, 25)
2. Minimum number of samples to be present at a node before splitting further (2, 10, 50, 90, 100, 150, 200)
3. Minimum decrease in impurity after every split (anything ranging from 0.0 to 1.0)

The early termination policy (BanditPolicy) specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer. In this case, it basically states to check the job every two iterations. If the primary metric (Accuracy) falls outside of the top 10% range, Azure ML terminate the job. This saves us from continuing to explore hyperparameters that don't show promise helping reach our target metric.

In HyperDriveConfig we specified Accuracy as the primary metric and our goal is to maximaize it. Hyperdrive will execute 40 runs at max to limit the consumption of resources.

Following screenshot displays the same:

![hyperdrive_config](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/hyperdrive_config.png)

### Results
HyperDrive tried out following different combinations before reaching the best set of values:

![hyperdrive_allmodels](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/hyperdrive_allmodels.png)

Optimal values of the hyperparameters tuned are as follows:
1. Maximum Depth: optimal value --max_depth = 20
2. Minimum number of samples: optimal value --min_samples_split = 2
3. Minimum decrease in impurity: optimal value --min_impurity_decrease = 0.03315527414519093

After trying out various combinations of the hyperparameters, maximum Accuracy achieved by HyperDrive is **0.9936**.

The run ID, accuracy, and the parameters of the model can be seen in the following screenshot:

![hyperdrive_bestmodel](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/hyperdrive_bestmodel.png)

We even used the `RunDetails` widget to monitor the run:

![hyperdrive_RunDetails](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/hyperdrive_RunDetails.png)

### Future Work
We can try following things for better results in future:
1. We need to tackle the class imbalance before feeding the data into our model. For this we can use resampling of the training data.
2. We can alter the parameter sampler for trying out new combinations of hyperparameter values.
3. In HyperDriveConfig, increase the value of max_total_runs.
4. Use a different train test split ratio.
5. And of course, a new classification algorithm with different hyperparameters can be used for the job at hand.

## Model Deployment
Since the AutoML found the better model we will deploy AutoML's model. To deploy this newly trained model we need to run the following piece of code as shown in the following screenshot.

![model_registration](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/model_registration.png)
![model_deployment](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/model_deployment.png)

After deployment, make sure the *Deployment state* is *Healthy* in the studio.

![deployed_service_1](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/deployed_service_1.png)

![deployed_service_2](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/deployed_service_2.png)

Now, there are two ways for testing our deployed model"
1. #### We send an HTTP request to the web service using `requests.post()` method.

![test_1](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/test_1.png)

2. #### We use the `service.run()` method.

![test_2](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/test_2.png)

This following screenshot shows the logs generated by the deployed service.

![service_logs](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/logs_output.png)

Once all said and done we should delete the deployed service and the compute target to avoid unnecessary consumption of resources.

![delete_service_compute](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/logs_output.png)

## Screen Recording
Here is the link to the screencast. This screencast demonstrate the following:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

[Let's go to the screencast!](https://youtu.be/ZmYkM15ZQ2k)

## Standout Suggestions
The one standout suggestion that I have attempted is to enable logging in the deployed service.

I used the following line of code in the notebook to enable ***Application Insights***.

![enable_app_insights](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/enable_app_insights_code.png)

This is the dashboard of Application Insights which makes analysis way easier.

![app_insights_dashboard](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/application_insights_dashboard.png)

## Additional Features
Apart from the standout suggestions mentioned in the project rubrics I took the liberty to try out the following two things.
1. #### Swagger Documentation
I have created the documentation for the REST endpoint of our deployed service. This can be reviewed in the following screenshot.

![swagger_1](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/swagger_1.png)
![swagger_2](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/swagger_2.png)
![swagger_3](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/swagger_3.png)

2. #### Benchmarking
This step will benchmark the endpoint using Apache Benchmark (ab). `benchmark.sh` contains one line of ab. The following screenshot shows Apache Benchmark (ab) running against the HTTP API using authentication keys to retrieve performance results.

![benchmarking](https://github.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/blob/main/screenshots/benchmarking.png)
