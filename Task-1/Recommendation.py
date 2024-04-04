from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf,col,when
import numpy as np


spark = SparkSession.builder.appName('Recommendation').getOrCreate()

sc = spark.sparkContext
sqlContext=SQLContext(sc)

ratings_df=spark.read.csv('s3://collinson-task-bucket/ratings.csv',inferSchema=True,header=True)
ratings_df.printSchema()


ratings_df.show()

movies_df=spark.read.csv('s3://collinson-task-bucket/movies.csv',inferSchema=True,header=True)
movies_df.printSchema()

movies_df.show()

links_df=spark.read.csv('s3://collinson-task-bucket/links.csv',inferSchema=True,header=True)
links_df.printSchema()

training_df,validation_df=ratings_df.randomSplit([0.8,0.2])

iterations=10
regularization_parameter=0.1
rank=4
error=[]
err=0

als = ALS(maxIter=iterations,regParam=regularization_parameter,rank=5,userCol="userId",itemCol="movieId",ratingCol="rating")
model=als.fit(training_df)
predictions=model.transform(validation_df)
new_predictions=predictions.filter(col('prediction')!=np.nan)
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
rmse=evaluator.evaluate(new_predictions)
print("Root Mean Square Error="+str(rmse))


for rank in range(4,10):
    als = ALS(maxIter=iterations,regParam=regularization_parameter,rank=rank,userCol="userId",itemCol="movieId",ratingCol="rating")
    model=als.fit(training_df)
    predictions=model.transform(validation_df)
    new_predictions=predictions.filter(col('prediction')!=np.nan)
    evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    rmse=evaluator.evaluate(new_predictions)
    print("Root Mean Square Error="+str(rmse))
    

predictions.show(n=10)

predictions.join(movies_df,"movieId").select("userId","title","genres","prediction").show(10)


for_one_user = predictions.filter(col("userId")==12).join(movies_df,"movieId").join(links_df,"movieId").select("userId","title","genres","tmdbId","prediction")
for_one_user.show(10)


userRecommends=model.recommendForAllUsers(5)
movieRecommends=model.recommendForAllItems(5)


userRecommends.printSchema()

userRecommends.select("userId","recommendations.movieId").show(10,False)

movieRecommends.printSchema()

movieRecommends.select("movieId","recommendations.userId").show(10,False)

users=ratings_df.select("userId").distinct().limit(5)
users.show()

userSubsetRecs = model.recommendForUserSubset(users,10)
userSubsetRecs.show()

userSubsetRecs.select("userId","recommendations.movieId").show(10,False)

movies=ratings_df.select("movieId").distinct().limit(5)
movies.show()

movieSubsetRecs = model.recommendForItemSubset(movies,10)
movieSubsetRecs.select("movieId","recommendations.userId").show(10,False)

movie_ids=[1580,3175,2366,1590]
user_ids=[543,543,543,543]
new_user_preds=sqlContext.createDataFrame(zip(movie_ids,user_ids),schema=['movieId','userId'])
new_predictions=model.transform(new_user_preds)
new_predictions.show()

