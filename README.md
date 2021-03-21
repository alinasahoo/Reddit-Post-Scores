# Predicting popularity of a Reddit post

<div align="center">
  <img src="/Images/reddit-logo.png" height="100" width="300">
</div>

### What is Reddit? <h3>
Reddit is a social sharing website where you could post links, pictures, text and other users can upvote or downvote aparticular post based on if they like the post or not. If the post gets a high upvote score, then the post moves up so that it is visible to evryone. Reddit is a huge site, but it's divided into thousands of smaller communities called subreddits. In this project, the posts from "Popular" subreddit were used to prepare the dataset.
<div align="center">
  <img src="/Images/Upvote_Downvote.png">
</div>
  
### About the Project: <h3>
This project is about predicting the popularity of a Reddit post. The popularity of a Reddit post is determined by the total votes or score it gets. Score is the result of upvotes and downvotes for a particular post. So, it was identified as a regression problem.

### About the Dataset: <h3>
The dataset for this project was created using web scarpping with the help of praw library. The features exracted are:
* Title - title of the post
* Gilded - rewarding a Reddit gold to the post
* Over_18 - True if the post has adult content else False
* Ups - no of upvotes for the post
* Downs - no of downvotes for the post
* Num_of_comments - no of comments for the post
* Upvote_ratio - upvote ratio of the post
* Score - total score (upvotes - downvotes)
  
Google Drive [link](https://drive.google.com/file/d/15nO0765lScyH17q-XvJ068hD7-spne0T/view?usp=sharing) for dataset (ScrappedPostsData.csv)

### Sentiment Analysis: <h3>
Extra features were added with the help of Sentiment analysis for the title of the post using vaderSentiment analyzer. We get 4 columns neg, neu, pos and compound. These features tell how negative or positive the statement is. These columns were combined to one column, Predited_value, using the compound score. 
  
positive sentiment: (compound score >= 0.05); neutral sentiment: (compound score > -0.05) and (compound score < 0.05);  negative sentiment: (compound score <= -0.05) 

### Text pre processing: <h3>
Text preprocessing was done for the title of the post by removing punctuations, stop words, and performing stemming and lemmatization. To convert the title to numeric form, Glove embedding was used. This gives a numeric vector for all the unique words in the text. These vectors have 100 dimensions. To get one vector representation for each title, weighted average method was used. The mean of all word vectors for a particular title is taken to form one vector so that the title is represented using one vector. One hot encoding for the Predicted_value and Over_18 columns was performed. 

### Machine Learning: <h3>
As it is a regression problem, regression models like Linear regression, Decision tree regressor, Random forest regressor, KNN regressor, Lasso, Ridge, ElasticNet and XGBoost regressor were used. These models were trained with 60% train data and prediction was done using 40% test data. The performance of these models was measured based on test accuracy. Out of all these models, XGBoost regressor and Random forest regressor performed well with around 50% accuracy on test dataset. 

### Deployment: <h3>
The application was deployed on Heroku. The application takes a Redidt post URl as input and the required features are extracted from the URL. The deployed application was tested with different Reddit post URLs. As the accuracy of the model is around 50%, the predictions were a little different from expected. In future work, more data can be used to train the model to get good accuracy.
  
Link to Deployed Application : https://reddit-post-score.herokuapp.com/

<div align="center">
  <img src="/Images/1.jpg" height="270" width="550"><img src="/Images/2.jpg" height="270" width="550">
</div>

### Installing required librarires: <h3>
* Installing __keras__ and __tenserflow__:
```
pip install keras==2.2.4
pip install tensorflow==2.0
```
* Installing nltk:
```
import nltk
nltk.download('popular')
```
* Installing xgboost:
```
pip install xgboost
```
* Installing vaderSentiment:
```
pip install vaderSentiment
```
* Installing praw:
```
pip install praw
```
