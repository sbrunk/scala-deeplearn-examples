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

import io.brunk.examples.IrisReader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ DenseLayer, OutputLayer }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{ Logger, LoggerFactory }

/**
  * A simple feed forward network for classifying the IRIS dataset in dl4j with a single hidden layer
  *
  * Based on
  * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java
  *
  * @author Sören Brunk
  */
object IrisMLP {
  private val log: Logger = LoggerFactory.getLogger(IrisMLP.getClass)

  def main(args: Array[String]): Unit = {

    val seed         = 1
    val numInputs    = 4
    val numHidden    = 4
    val numOutputs   = 3
    val learningRate = 0.1
    val iterations   = 1000

    val testAndTrain = IrisReader.readData()

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(Activation.RELU)
      .weightInit(WeightInit.XAVIER)
      .learningRate(learningRate)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHidden).build())
      .layer(1,
             new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
               .activation(Activation.SOFTMAX)
               .nIn(numHidden)
               .nOut(numOutputs)
               .build())
      .backprop(true)
      .pretrain(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100)) // print out scores every 100 iterations

    log.info("Running training")
    model.fit(testAndTrain.getTrain)
    log.info("Training finished")

    log.info(s"Evaluating model on ${testAndTrain.getTest.getLabels.rows()} examples")
    val evaluator = new Evaluation(numOutputs)
    val output    = model.output(testAndTrain.getTest.getFeatureMatrix)
    evaluator.eval(testAndTrain.getTest.getLabels, output)
    println(evaluator.stats)
  }
}
