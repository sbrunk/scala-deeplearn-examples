/*
 * Copyright 2017 Sören Brunk
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright 2017 Sören Brunk
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.brunk.examples

import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.{DataType, _}
import org.platanios.tensorflow.api.ops.NN.ValidConvPadding
import org.slf4j.LoggerFactory
import better.files._

import scala.util.Random.shuffle
import org.platanios.tensorflow.api.ops.io.data.{Dataset, TensorSlicesDataset}

/**
  * CNN image classification examples
  *
  * @author Sören Brunk
  */
object SimpleCNN {
  private[this] val logger = Logger(LoggerFactory.getLogger(getClass))

  def main(args: Array[String]): Unit = {

    val batchSize    = 32
    val numEpochs    = 10

    val dataDir = File(args(0))
    val imgClasses = dataDir.list.filter(_.isDirectory).toVector.sortBy(_.name)
    val labelForClass = imgClasses.zipWithIndex.toMap

    val shuffledSamples =  {
      // in case of different numbers of samples per class, get the smallest one
      val numSamplesPerClass = imgClasses.map(_.glob("*.jpg").size).min
      imgClasses.flatMap { imgClass =>
        shuffle(imgClass.children.toVector)
          .take(numSamplesPerClass)       // balance samples to have the same number for each class
          .map(imgFile => (imgFile, labelForClass(imgClass)))
      }
    }

    val numSamples = shuffledSamples.size
    logger.info("Number of samples {}", numSamples)
    val numTrainSamples = (0.8 * numSamples).toInt
    val numTestSamples = (0.2 * numSamples).toInt

    val (filenames, labels) = {
      val (filenames, labels) = shuffledSamples.unzip
      logger.info(Tensor(filenames.map(filename => filename.pathAsString)).squeeze(Seq(0)).summarize())
      logger.info(Tensor(labels).squeeze(Seq(0)).summarize())
      (Tensor(filenames.map(filename => filename.pathAsString)).squeeze(Seq(0)), Tensor(UINT8, labels).squeeze(Seq(0)))
    }

    val dataSet: Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] =
      tf.data.TensorSlicesDataset(filenames)
      .map { filename =>
        val rawImage = tf.data.readFile(filename)
        val image = tf.image.decodeJpeg(rawImage)
        image
        // TODO resize image
      }.zip(tf.data.TensorSlicesDataset(labels))


    logger.info(dataSet.outputShapes._1.toString())

    val trainData = dataSet
      .take(numTrainSamples)
      .repeat()
      //.shuffle(10000)
      .batch(batchSize)
      .prefetch(10)
    val evalTrainData = dataSet.batch(1000).prefetch(10)
    val evalTestData = dataSet.drop(numTrainSamples).batch(1000).prefetch(10)

    logger.info(trainData.outputShapes._1.toString())


    // define the neural network architecture
    val input = tf.learn.Input(UINT8, Shape(-1, dataSet.outputShapes._1(0), dataSet.outputShapes._1(1), dataSet.outputShapes._1(2))) // type and shape of images
    val trainInput = tf.learn.Input(UINT8, Shape(-1)) // type and shape of labels

    val layers =
      tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Layer_0/Conv2D", Shape(3, 3, 3, 32), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_0/Bias") >>
      tf.learn.ReLU("Layer_0/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_0/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 1, stride2 = 1, padding = ValidConvPadding)
      tf.learn.Conv2D("Layer_1/Conv2D", Shape(3, 3, 32, 64), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_1/Bias") >>
      tf.learn.ReLU("Layer_1/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_1/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 1, stride2 = 1, padding = ValidConvPadding)
      tf.learn.Conv2D("Layer_2/Conv2D", Shape(3, 3, 64, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_2/Bias") >>
      tf.learn.ReLU("Layer_2/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_2/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 1, stride2 = 1, padding = ValidConvPadding)
      tf.learn.Conv2D("Layer_3/Conv2D", Shape(3, 3, 128, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_3/Bias") >>
      tf.learn.ReLU("Layer_3/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_3/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 1, stride2 = 1, padding = ValidConvPadding)
      tf.learn.Dropout("Layer_3/Dropout", keepProbability = 0.5f)
      tf.learn.Flatten("Layer_3/Flatten")
      tf.learn.Linear("Layer_4/Linear", units = 512) >> tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
      tf.learn.Linear("OutputLayer/Linear", 1)

    val trainInputLayer = tf.learn.Cast("TrainInput/Cast", INT64) // cast labels to long

    val loss = tf.learn.SigmoidCrossEntropy("Loss/CrossEntropy") >> tf.learn.Mean("Loss/Mean") >>
      tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.AdaGrad(0.1f)

    val model = tf.learn.Model.supervised(input, layers, trainInput, trainInputLayer, loss, optimizer)

//    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
//        tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss"
//    val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    val summariesDir = Paths.get("temp/simple-cnn")
    //val accMetric = tf.metrics.MapMetric(
    //  (v: (Output, Output)) => (v._1.argmax(-1), v._2), tf.metrics.Accuracy())
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some((60000/batchSize)*numEpochs)), // due to a bug, we can't use epochs directly
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        //tf.learn.Evaluator(
        //  log = true, datasets = Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
        //  metrics = Seq(accMetric), trigger = tf.learn.StepHookTrigger(1000), name = "Evaluator"),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))

    // train the model
    estimator.train(() => trainData)

    //def accuracy(images: Tensor, labels: Tensor): Float = {
    //  val predictions = estimator.infer(() => images)
    //  predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
    //}

    // evaluate model performance
    //logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    //logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}
