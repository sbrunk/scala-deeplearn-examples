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
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.image.MNISTLoader
import org.platanios.tensorflow.data.image.MNISTLoader.FASHION_MNIST
import org.slf4j.LoggerFactory

/** Simple multilayer perceptron for classifying handwritten digits from the Fashion MNIST dataset
  *
  * Implemented using TensorFlow for Scala based on the example from
  * https://github.com/eaplatanios/tensorflow_scala/blob/0b7ca14de53935a34deac29802d085729228c4fe/examples/src/main/scala/org/platanios/tensorflow/examples/MNIST.scala
  *
  * @author Sören Brunk
  */
object FashionMnistMLP {
  private[this] val logger = Logger(LoggerFactory.getLogger(FashionMnistMLP.getClass))

  def main(args: Array[String]): Unit = {

    val batchSize    = 2048
    val numEpochs    = 500

    // download and load the MNIST images as tensors
    val dataSet = MNISTLoader.load(Paths.get("datasets/Fashion-MNIST"), FASHION_MNIST)
    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
    val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
    val testImages = tf.data.TensorSlicesDataset(dataSet.testImages)
    val testLabels = tf.data.TensorSlicesDataset(dataSet.testLabels)
    val trainData =
      trainImages.zip(trainLabels)
          .repeat()
          .shuffle(60000)
          .batch(batchSize)
          .prefetch(10)
    val evalTrainData = trainImages.zip(trainLabels).batch(1000).prefetch(10)
    val evalTestData = testImages.zip(testLabels).batch(1000).prefetch(10)

    // define the neural network architecture
    val input = tf.learn.Input(UINT8, Shape(-1, 28, 28))          // type and shape of our input images
    val labelInput = tf.learn.Input(UINT8, Shape(-1))             // type and shape of our labels

    val layer = tf.learn.Flatten("Input/Flatten") >>              // flatten the images into a single vector
      tf.learn.Cast("Input/Cast", FLOAT32) >>                     // cast input to float
      tf.learn.Linear("Layer_1/Linear", units = 512) >>           // hidden layer
      tf.learn.ReLU("Layer_1/ReLU", 0.1f) >>                            // hidden layer activation
      tf.learn.Linear("OutputLayer/Linear", units = 10)           // output layer

    val trainInputLayer = tf.learn.Cast("TrainInput/Cast", INT64) // cast labels to long

    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>   // loss/error function
        tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.Adam(learningRate = 0.001)          // the optimizer updates our weights

    val model = tf.learn.Model.supervised(input, layer, labelInput, trainInputLayer, loss, optimizer)

    val summariesDir = Paths.get("temp/fashion-mnist-mlp")
    val accMetric = tf.metrics.MapMetric(
      (v: (Output, Output)) => (v._1.argmax(-1), v._2), tf.metrics.Accuracy())
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some((60000/batchSize)*numEpochs)), // due to a bug, we can't use epochs directly
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.Evaluator(
          log = true, datasets = Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
          metrics = Seq(accMetric), trigger = tf.learn.StepHookTrigger(500), name = "Evaluator", summaryDir = summariesDir),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(500))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))

    // train the model
    estimator.train(() => trainData)

    def accuracy(images: Tensor, labels: Tensor): Float = {
      val predictions = estimator.infer(() => images)
      predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
    }

    // evaluate model performance
    logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}
