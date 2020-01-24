from pyspark import SparkContext, SparkConf
from pyspark.shell import spark
from pyspark.sql import SQLContext, SparkSession
import numpy as np
from pyspark.sql import DataFrame
# from pyspark.sql import SQLContext
from pyspark.sql.functions import when
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.types import StructType, StructField, NumericType
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
import time


def diabetes():
    # sc = SparkContext().getOrCreate()
    sc = SparkSession.builder.getOrCreate()
    # url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
    raw_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(
        "E:\Projects\PythonProjects\Sparkando\diabetes.csv")
    columns = raw_data.columns
    print(columns)
    # sc.addFile(url)
    # raw_data.describe().select("Summary", "Pregnancies", "Glucose", "BloodPressure").show()
    # raw_data.describe().select("Summary", "SkinThickness", "Insulin").show()
    # raw_data.describe().select("Summary", "BMI", "DiabetesPedigreeFunction", "Age").show()
    raw_data = raw_data.withColumn("Glucose", when(raw_data.Glucose == 0, np.nan).otherwise(raw_data.Glucose))
    raw_data = raw_data.withColumn("BloodPressure", when(raw_data.BloodPressure == 0, np.nan).otherwise(raw_data.BloodPressure))
    raw_data = raw_data.withColumn("SkinThickness", when(raw_data.SkinThickness == 0, np.nan).otherwise(raw_data.SkinThickness))
    raw_data = raw_data.withColumn("BMI", when(raw_data.BMI == 0, np.nan).otherwise(raw_data.BMI))
    raw_data = raw_data.withColumn("Insulin", when(raw_data.Insulin == 0, np.nan).otherwise(raw_data.Insulin))
    raw_data.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5)
    imputer = Imputer(inputCols=["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"],
                      outputCols=["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"])
    model = imputer.fit(raw_data)
    raw_data = model.transform(raw_data)
    raw_data.show(5)

    # combine all the features in one single feature vector.
    cols = raw_data.columns
    cols.remove("Outcome")
    # Let us import the vector assembler
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    # Now let us use the transform method to transform our dataset
    raw_data = assembler.transform(raw_data)
    print(type(raw_data))
    raw_data.select("features").show(truncate=False)

    standardscaler = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    raw_data = standardscaler.fit(raw_data).transform(raw_data)
    raw_data.select("features", "Scaled_features").show(5)

    train, test = raw_data.randomSplit([0.8, 0.2], seed=12345)

    dataset_size = float(train.select("Outcome").count())
    numPositives = train.select("Outcome").where('Outcome == 1').count()
    per_ones = (float(numPositives) / float(dataset_size)) * 100
    numNegatives = float(dataset_size - numPositives)
    print('The number of ones are {}'.format(numPositives))
    print('Percentage of ones are {}'.format(per_ones))

    BalancingRatio = numNegatives / dataset_size
    print('BalancingRatio = {}'.format(BalancingRatio))

    train = train.withColumn("classWeights", when(train.Outcome == 1, BalancingRatio).otherwise(1 - BalancingRatio))
    train.select("classWeights").show(5)

    css = ChiSqSelector(featuresCol='Scaled_features', outputCol='Aspect', labelCol='Outcome', fpr=0.05)
    train = css.fit(train).transform(train)
    test = css.fit(test).transform(test)
    test.select("Aspect").show(5, truncate=False)

    lr = LogisticRegression(labelCol="Outcome", featuresCol="Aspect", weightCol="classWeights", maxIter=10)
    model = lr.fit(train)
    predict_train = model.transform(train)
    predict_test = model.transform(test)
    predict_test.select("Outcome", "prediction").show(10)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Outcome")
    predict_test.select("Outcome", "rawPrediction", "prediction", "probability").show(5)
    print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
    print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))

    # lnr = GeneralizedLinearRegression(labelCol="Outcome", featuresCol="Aspect", weightCol="classWeights", maxIter=10)
    # model = lnr.fit(train)
    # predict_train = model.transform(train)
    # predict_test = model.transform(test)
    # predict_test.select("Outcome", "prediction").show(10)
    #
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Outcome")
    # predict_test.select("Outcome", "rawPrediction", "prediction", "probability").show(5)
    # print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
    # print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))

    # rf = RandomForestClassifier(labelCol="Outcome", featuresCol="Aspect", numTrees=10)
    # model = rf.fit(train)
    # predict_train = model.transform(train)
    # predict_test = model.transform(test)
    # predict_test.select("Outcome", "prediction").show(10)
    #
    # predictions = model.transform(test)
    # evaluator = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="Aspect", metricName="accuracy")
    # accuracy = evaluator.evaluate(predictions)
    # print("Test Error = %g" % (1.0 - accuracy))
    # sqlContext = SQLContext(sc)
    sc.stop()


def isSick(x):
    if x in (3, 7):
        return 0
    else:
        return 1


def classify():
    cols = ['age',
            'sex',
            'chest pain',
            'resting blood pressure',
            'serum cholesterol',
            'fasting blood sugar',
            'resting electrocardiographic results',
            'maximum heart rate achieved',
            'exercise induced angina',
            'ST depression induced by exercise relative to rest',
            'the slope of the peak exercise ST segment',
            'number of major vessels ',
            'thal',
            'last']

    data = pd.read_csv('E:\Projects\PythonProjects\Sparkando\heart.csv', delimiter=' ', names=cols)
    data = data.iloc[:, 0:13]
    data['label'] = data['thal'].apply(isSick)
    df = spark.createDataFrame(data)

    features = ['age',
                'sex',
                'chest pain',
                'resting blood pressure',
                'serum cholesterol',
                'fasting blood sugar',
                'resting electrocardiographic results',
                'maximum heart rate achieved',
                'exercise induced angina',
                'ST depression induced by exercise relative to rest',
                'the slope of the peak exercise ST segment',
                'number of major vessels ']

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    raw_data = assembler.transform(df)
    raw_data.select("features").show(truncate=False)

    standardscaler = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    raw_data = standardscaler.fit(raw_data).transform(raw_data)
    raw_data.select("features", "Scaled_features").show(5)

    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

    training, test = raw_data.randomSplit([0.5, 0.5], seed=12345)
    from pyspark.ml.classification import LogisticRegression

    # ----------------------------- LOGISTIC REGRESSION -----------------------------
    # lr = LogisticRegression(labelCol="label", featuresCol="Scaled_features", maxIter=100)
    # model = lr.fit(training)
    # plt.figure(figsize=(5, 5))
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.plot(model.summary.roc.select('FPR').collect(),
    #          model.summary.roc.select('TPR').collect())
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.show()
    # predict_train = model.transform(training)
    # predict_test = model.transform(test)
    # predict_test.select("label", "prediction").show(10)
    # print("Multinomial coefficients: " + str(model.coefficientMatrix))
    # print("Multinomial intercepts: " + str(model.interceptVector))
    # # import pyspark.sql.functions as F
    # # check = predict_test.withColumn('correct', F.when(F.col('isSick') == F.col('prediction'), 1).otherwise(0))
    # # check.groupby("correct").count().show()
    # evaluator = BinaryClassificationEvaluator()
    # print("Test Area Under ROC: " + str(evaluator.evaluate(predict_test, {evaluator.metricName: "areaUnderROC"})))

    # ----------------------------- RANDOM FOREST -----------------------------
    rf = RandomForestClassifier(labelCol="label", featuresCol="Scaled_features", numTrees=200)
    model = rf.fit(training)
    predict_train = model.transform(training)
    predict_test = model.transform(test)
    predict_test.select("label", "prediction").show(10)

    # print("Multinomial coefficients: " + str(model.coefficientMatrix))
    # print("Multinomial intercepts: " + str(model.interceptVector))
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predict_test, {evaluator.metricName: "areaUnderROC"})))

    return 0


def main():
    classify()
    # diabetes()


if __name__ == "__main__":
    main()
