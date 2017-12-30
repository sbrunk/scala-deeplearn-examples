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
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.models.Sequential
import org.deeplearning4j.scalnet.optimizers.SGD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.JavaConverters.asScalaIteratorConverter


/** Simple multilayer perceptron for classifying handwritten digits from the MNIST dataset
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
    val numHidden    = 128
    val numOutputs   = 10      // digits from 0 to 9
    val learningRate = 0.01
    val batchSize    = 128
    val numEpochs    = 10

    // download and load the MNIST images as tensors
    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, seed)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, seed)

    // define the neural network architecture
    val model: Sequential = Sequential(rngSeed = seed)
    model.add(Dense(nOut = numHidden, nIn = numInputs, weightInit = WeightInit.XAVIER,  activation = "relu"))
    model.add(Dense(nOut = numOutputs, weightInit = WeightInit.XAVIER, activation = "softmax"))
    model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD, optimizer = SGD(learningRate, momentum = 0,
      nesterov = true))

    // train the model
    model.fit(mnistTrain, nbEpoch = numEpochs, List(new ScoreIterationListener(100)))

    // evaluate model performance
    val evaluator = new Evaluation(numOutputs)
    for (dataSet <- mnistTest.asScala) {
      val output = model.predict(dataSet)
      evaluator.eval(dataSet.getLabels, output)
    }
    log.info(evaluator.stats())
  }
}
