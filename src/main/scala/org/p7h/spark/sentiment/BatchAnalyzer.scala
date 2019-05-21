package org.p7h.spark.sentiment

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql._
import org.apache.spark.sql.types
import org.apache.spark.{SparkConf, SparkContext}
import org.p7h.spark.sentiment.utils._
import org.p7h.spark.sentiment.mllib._
import java.text.SimpleDateFormat



object BatchAnalyzer {

  def main(args: Array[String]) {
    val sc = createSparkContext()
    LogUtils.setLogLevels(sc)
    val naiveBayesModel: NaiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.naiveBayesModelPath)
    Console.println("NaiveBayesModel Caricato");
    val stopWordsList = sc.broadcast(StopwordsLoader.loadStopWords(PropertiesLoader.nltkStopWords))
		batchAnalyze(sc, stopWordsList, naiveBayesModel)
  }

  /**
    * Remove new line characters.
    *
    * @param tweetText -- Complete text of a tweet.
    * @return String with new lines removed.
    */
  def replaceNewLines(tweetText: String): String = {
    tweetText.replaceAll("\n", "")
  }

  /**
    * Create SparkContext.
    * Future extension: enable checkpointing to HDFS [is it really reqd??].
    *
    * @return SparkContext
    */
  def createSparkContext(): SparkContext = {
    val conf = new SparkConf()
      .setAppName(this.getClass.getSimpleName)
      .set("spark.serializer", classOf[KryoSerializer].getCanonicalName)
    val sc = SparkContext.getOrCreate(conf)
    sc
  }


  /**
    *
    * @param sc            -- Spark Context.
    * @param stopWordsList -- Broadcast variable for list of stop words to be removed from the tweets.
    */
  def batchAnalyze(sc: SparkContext, stopWordsList: Broadcast[List[String]], naiveBayesModel: NaiveBayesModel): Unit = {

    val tweetsDF: DataFrame = loadSentiment140File(sc, PropertiesLoader.batchTestDataAbsolutePath)

    val sentimentAnalysisRDD = tweetsDF.select("polarity","user", "status").rdd.map {
      case Row(polarity: Int, user: String, tweet: String) =>
        val tweetText = replaceNewLines(tweet)
        val tweetInWords: Seq[String] = MLlibSentimentAnalyzer.getBarebonesTweetText(tweetText, stopWordsList.value)
        (polarity.toDouble,naiveBayesModel.predict(MLlibSentimentAnalyzer.transformFeatures(tweetInWords)),user,tweetText)
    }

    //Salvo elementi RDD nel file txt
    //sentimentAnalysisRDD.saveAsTextFile(PropertiesLoader.tweetsClassifiedPath)
    //Console.println("Elementi salvati in file.txt")
    saveClassifiedTweets(sentimentAnalysisRDD, PropertiesLoader.tweetsClassifiedPath)
    
        Console.println("Result prediction saved!");

  }

       /**
    * Saves the classified tweets to the csv file.
    * Uses DataFrames to accomplish this task.
    *
    * @param rdd                  tuple with Tweet Polarity, Tweet Text
    * @param tweetsClassifiedPath Location of saving the data.
    */
  def saveClassifiedTweets(rdd: RDD[(Double, Double,String, String)], tweetsClassifiedPath: String) = {
  val dir = new String("results")
    val sqlContext = SQLContextSingleton.getInstance(rdd.sparkContext)
    import sqlContext.implicits._
    val classifiedTweetsDF = rdd.toDF("Polarity", "MLLib Polarity","User", "Text")
    classifiedTweetsDF.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ",")
      // Compression codec to compress when saving to file.
      .option("codec", classOf[GzipCodec].getCanonicalName)
      // Will it be good if DF is partitioned by Sentiment value? Probably does not make sense.
      //.partitionBy("Sentiment")
      // TODO :: Bug in spark-csv package :: Append Mode does not work for CSV: https://github.com/databricks/spark-csv/issues/122
      //.mode(SaveMode.Append)
      .save(tweetsClassifiedPath + dir)
  }  



  /**
    * Loads the Sentiment140 file from the specified path using SparkContext.
    *
    * @param sc                   -- Spark Context.
    * @param sentiment140FilePath -- Absolute file path of Sentiment140.
    * @return -- Spark DataFrame of the Sentiment file with the tweet text and its polarity.
    */
  def loadSentiment140File(sc: SparkContext, sentiment140FilePath: String): DataFrame = {
    val sqlContext = SQLContextSingleton.getInstance(sc)
    val tweetsDF = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .load(sentiment140FilePath)
      .toDF("polarity", "id", "date", "query", "user", "status")

    // Drop the columns we are not interested in.
    tweetsDF.drop("id").drop("date").drop("query")
  }
	

	
}
