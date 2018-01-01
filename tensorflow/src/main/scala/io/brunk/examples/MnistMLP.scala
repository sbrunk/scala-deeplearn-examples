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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.image.MNISTLoader
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import java.nio.file.Paths

import org.platanios.tensorflow.api.ops.variables.GlorotUniformInitializer

/** Simple multilayer perceptron for classifying handwritten digits from the MNIST dataset
  *
  * Implemented using TensorFlow for Scala based on the example from
  * https://github.com/eaplatanios/tensorflow_scala/blob/0b7ca14de53935a34deac29802d085729228c4fe/examples/src/main/scala/org/platanios/tensorflow/examples/MNIST.scala
  *
  * @author Sören Brunk
  */
object MnistMLP {
  private[this] val logger = Logger(LoggerFactory.getLogger(MnistMLP.getClass))

  def main(args: Array[String]): Unit = {

    val seed         = 1       // for reproducibility
    val numInputs    = 28 * 28
    val numHidden    = 512
    val numOutputs   = 10      // digits from 0 to 9
    val learningRate = 0.01
    val batchSize    = 128
    val numEpochs    = 10

    // download and load the MNIST images as tensors
    val dataSet = MNISTLoader.load(Paths.get("datasets/MNIST"))
    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
    val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
    val testImages = tf.data.TensorSlicesDataset(dataSet.testImages)
    val testLabels = tf.data.TensorSlicesDataset(dataSet.testLabels)
    val trainData =
      trainImages.zip(trainLabels)
          .repeat()
          .shuffle(10000)
          .batch(batchSize)
          .prefetch(10)
    val evalTrainData = trainImages.zip(trainLabels).batch(1000).prefetch(10)
    val evalTestData = testImages.zip(testLabels).batch(1000).prefetch(10)

    // define the neural network architecture
    val input = tf.learn.Input(UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2)))
    val trainInput = tf.learn.Input(UINT8, Shape(-1))
    val layer = tf.learn.Flatten("Input/Flatten") >>
        tf.learn.Cast("Input/Cast", FLOAT32) >>
        tf.learn.Linear("Layer_1/Linear", numHidden, weightsInitializer = GlorotUniformInitializer()) >> tf.learn.ReLU("Layer_1/ReLU", 0.01f) >>
        tf.learn.Linear("OutputLayer/Linear", numOutputs, weightsInitializer = GlorotUniformInitializer())
    val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)
    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
        tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.Adam(learningRate)
    val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    val summariesDir = Paths.get("temp/mnist-mlp")
    val accMetric = tf.metrics.MapMetric(
      (v: (Output, Output)) => (v._1.argmax(-1), v._2), tf.metrics.Accuracy())
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some((60000/batchSize)*numEpochs)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.Evaluator(
          log = true, data = () => evalTrainData, metrics = Seq(accMetric),
          trigger = tf.learn.StepHookTrigger(1000), name = "TrainEvaluation"),
        tf.learn.Evaluator(
          log = true, data = () => evalTestData, metrics = Seq(accMetric),
          trigger = tf.learn.StepHookTrigger(1000), name = "TestEvaluation"),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
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
