
## Project structure
* This project built from 3 file types: py, ipynb, csv
* For each ipynb there is equivalent py file.
* The ipynb import functionality from the py files. This directionality built in class structure.
    - Cause this is not production code, all the method defined to be static methods (Not good practice but easier for that mission)
* Project flow:
    - a-data-overview  - `General overview, data tests and a future action items`
    - b-feature-engineering - `Aggregating data - labeling and feature extraction`
    - c-feature-selection - `Picking the most influantal features - feature sensativity, feature importence, Generalization penalty (Lasso)`
    - d-data-enrichment - `As part of targeting our data bais issue - resample and adding nois, grouping (previous_policies and k means) and shuffeling features`
    - e-train-and-evaluate - `Combine all sections in order to create the best classification model`
    - f_production_training - `Train model (grid search, cross validation) for production with best reampeling and feature selection methods`
    - g_production_serving - `Load production model, predict labels for requests`

## Bottom line
* Our data is extremely biased - 2.5% of our users are labeled as 1.
* F1 is the main measurement which we use to investigate our model quality. (which combine recall and precision at one), The highest f1 we get to is 0.11.
* Corresponded to product needs we can changes f1 to fi and emphasis recall/precision importance.
* Resampeling and shuffling feature across the label-1 group is one of the techniques which helped us to target the bais issue.
* Boosting combining exponential loss looks like a way that can help in farther research, and that because it raises the weight of the misclassified points.


## Next steps that should be taken in a farther research
* 1) Dive into several algorithms ,cross validate and search for optimal params - 
     Since we focused was on feature selection, feature engineering  and the bais issu, the following code not contain deep actions of the above
* 2) Resample points which are failed to classify (here is sampled all one's evenly).
* 3) Clean exceptional points - for example points which classed as 1 with non frequent feature.
* 4) Investigating correlations and interactions between our features (Tried to do it partially by using rbf kernel)
* 5) Try to bin some of our continues features.
* 6) Use another unsupervised learning techniques (or even k-mean with more higher k ) for grouping and shuffling
* 7) Add charts of which show connections between variables and labels, errors, qq-plot etc'..
* 8) Play with train test splits: several sizes/split on users (here i split on user and policy id)
* 9) Import another features which can be easily joind on postal code (meidan/mean ages, ethnic proportions groups etc.)
* 10) Adding times dimension for example : policy age, time in app etc'...
* 11) Investigate another boosting algorithms as lightGBM.
* 12) Investigate another aggressive loss functions as exponential loss in order to increasing the weights of the 1's label in our data.
