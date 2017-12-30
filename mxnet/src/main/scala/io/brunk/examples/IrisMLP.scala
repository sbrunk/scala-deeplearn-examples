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

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.optimizer.SGD

object IrisMLP {

  def main(args: Array[String]): Unit = {

    val numInputs    = 4
    val numHidden    = 10
    val numOutputs   = 3
    val learningRate = 0.1f
    val iterations   = 1000
    val trainSize    = 100
    val testSize     = 50

    val batchSize    = 50
    val epochs = (iterations / (batchSize.toFloat / trainSize)).toInt

    // The mxnet Scala IO API does not support shuffling so we just read the csv using plain Scala
    val source = scala.io.Source.fromFile("data/iris.csv")
    val rows = source.getLines().drop(1).map { l =>
      val columns = l.split(",").map(_.toFloat)
      new {
        val features = columns.take(4)
        val labels = columns(4)
      }
    }.toBuffer
    val shuffled = scala.util.Random.shuffle(rows).toArray
    val trainData = shuffled.take(trainSize)
    val testData = shuffled.drop(trainSize)
    val trainFeatures = NDArray.array(trainData.flatMap(_.features), Shape(trainSize, numInputs))
    val trainLabels = NDArray.array(trainData.map(_.labels), Shape(trainSize))
    val testFeatures = NDArray.array(testData.flatMap(_.features), Shape(testSize, numInputs))
    val testLabels = NDArray.array(testData.map(_.labels), Shape(testSize))


    val trainDataIter = new NDArrayIter(data = IndexedSeq(trainFeatures), label = IndexedSeq(trainLabels), dataBatchSize = 50)
    val testDataIter = new NDArrayIter(data = IndexedSeq(testFeatures), label = IndexedSeq(testLabels), dataBatchSize = 50)

    // Define the network architecture
    val data = Symbol.Variable("data")
    val label = Symbol.Variable("label")
    val l1 = Symbol.FullyConnected(name = "l1")()(Map("data" -> data, "num_hidden" -> numHidden))
    val a1 = Symbol.Activation(name = "a1")()(Map("data" -> l1, "act_type" -> "relu"))
    val l2 = Symbol.FullyConnected(name = "l2")()(Map("data" -> a1, "num_hidden" -> numOutputs))
    val out = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> l2, "label" -> label))

    // Create and train a model
    val model = FeedForward.newBuilder(out)
      .setContext(Context.cpu()) // change to gpu if available
      .setNumEpoch(epochs)
      .setOptimizer(new SGD(learningRate = learningRate))
      .setTrainData(trainDataIter)
      .setEvalData(testDataIter)
      .build()
  }

}
