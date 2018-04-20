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

package io.brunk.examples.scalnet

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.models.Sequential
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters.asScalaIteratorConverter


/** Simple multilayer perceptron for classifying handwritten digits from the MNIST dataset.
  *
  * Implemented using ScalNet.
  *
  * @author Sören Brunk
  */
object MnistMLP {
  private val log: Logger = LoggerFactory.getLogger(MnistMLP.getClass)

  def main(args: Array[String]): Unit = {

    val seed         = 1       // for reproducibility
    val numInputs    = 28 * 28
    val numHidden    = 512     // size (number of neurons) in our hidden layer
    val numOutputs   = 10      // digits from 0 to 9
    val learningRate = 0.01
    val batchSize    = 128
    val numEpochs    = 10

    // download and load the MNIST images as tensors
    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, seed)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, seed)

    // define the neural network architecture
    val model: Sequential = Sequential(rngSeed = seed)
    model.add(Dense(nOut = numHidden, nIn = numInputs, weightInit = WeightInit.XAVIER, activation = Activation.RELU))
    model.add(Dense(nOut = numOutputs, weightInit = WeightInit.XAVIER, activation = Activation.RELU))
    model.compile(lossFunction = LossFunction.MCXENT, updater = Updater.SGD) // TODO how do we set the learning rate?

    // train the model
    model.fit(mnistTrain, nbEpoch = numEpochs, List(new ScoreIterationListener(100)))

    // evaluate model performance
    def accuracy(dataSet: DataSetIterator): Double = {
      val evaluator = new Evaluation(numOutputs)
      dataSet.reset()
      for (dataSet <- dataSet.asScala) {
        val output = model.predict(dataSet)
        evaluator.eval(dataSet.getLabels, output)
      }
      evaluator.accuracy()
    }

    log.info(s"Train accuracy = ${accuracy(mnistTrain)}")
    log.info(s"Test accuracy = ${accuracy(mnistTest)}")
  }
}
