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

import io.brunk.examples.IrisReader
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.models.Sequential
import org.deeplearning4j.scalnet.optimizers.SGD
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{ Logger, LoggerFactory }

/**
  * A simple feed forward network (one hidden layer) for classifying the IRIS dataset
  * implemented using ScalNet.
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

    val testAndTrain  = IrisReader.readData()
    val trainList     = testAndTrain.getTrain.asList()
    val trainIterator = new ListDataSetIterator(trainList, trainList.size)

    val model = Sequential(rngSeed = seed)
    model.add(
      Dense(numHidden, nIn = numInputs, weightInit = WeightInit.XAVIER, activation = "relu")
    )
    model.add(
      Dense(numOutputs, weightInit = WeightInit.XAVIER, activation = "softmax")
    )

    // Due to a bug in ScalNet 0.9.1 we have to enable nesterov.
    // Setting momentum to 0 it should disable it again though.
    model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD,
                  optimizer = SGD(learningRate, momentum = 0, nesterov = true))

    log.info("Running training")
    model.fit(iter = trainIterator,
              nbEpoch = iterations,
              listeners = List(new ScoreIterationListener(100)))
    log.info("Training finished")

    log.info(s"Evaluating model on ${testAndTrain.getTest.getLabels.rows()} examples")
    val evaluator        = new Evaluation(numOutputs)
    val output: INDArray = model.predict(testAndTrain.getTest.getFeatureMatrix)
    evaluator.eval(testAndTrain.getTest.getLabels, output)
    log.info(evaluator.stats())

  }
}
