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

package io.brunk.examples.dl4j

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters.asScalaIteratorConverter


/** Simple multilayer perceptron for classifying handwritten digits from the MNIST dataset.
  *
  * Implemented using DL4J based on the Java example from
  * https://github.com/deeplearning4j/dl4j-examples/blob/dfcf71d75fff956db53a93b09b560d53e3da4638/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java
  *
  * @author Sören Brunk
  */
object MnistMLP {
  private val log = LoggerFactory.getLogger(MnistMLP.getClass)

  def main(args: Array[String]): Unit = {

    val seed         = 1       // for reproducibility
    val numInputs    = 28 * 28
    val numHidden    = 128
    val numOutputs   = 10      // digits from 0 to 9
    val learningRate = 0.01
    val batchSize    = 128
    val numEpochs    = 10

    // download and load the MNIST images as tensors
    val mnistTrain = new MnistDataSetIterator(batchSize, true, seed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, seed)

    // define the neural network architecture
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(learningRate)
      .list // builder for creating stacked layers
      .layer(0, new DenseLayer.Builder() // define the hidden layer
        .nIn(numInputs)
        .nOut(numHidden)
        .activation(Activation.RELU)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // define the output layer
        .nIn(numHidden)
        .nOut(numOutputs)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))   // print the score every 100th iteration

    // train the model
    for (_ <- 0 until numEpochs) {
      model.fit(mnistTrain)
    }

    // evaluate model performance
    val evaluator = new Evaluation(numOutputs)
    for (dataSet <- mnistTest.asScala) {
      val output = model.output(dataSet.getFeatureMatrix)
      evaluator.eval(dataSet.getLabels, output)
    }
    log.info(evaluator.stats)
  }
}
