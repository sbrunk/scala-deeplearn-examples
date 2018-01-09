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
import ml.dmlc.mxnet.optimizer.SGD

/** Simple multilayer perceptron for classifying handwritten digits from the MNIST dataset.
  *
  * Implemented using MXNet.
  * Based on https://mxnet.incubator.apache.org/tutorials/scala/mnist.html
  *
  * @author Sören Brunk
  */
object MnistMLP {

  def main(args: Array[String]): Unit = {

    val numHidden    = 512     // size (number of neurons) of our hidden layer
    val numOutputs   = 10      // digits from 0 to 9
    val learningRate = 0.01f
    val batchSize    = 128
    val numEpochs    = 10

    // load the MNIST images as tensors
    val trainDataIter = IO.MNISTIter(Map(
      "image" -> "mnist/train-images-idx3-ubyte",
      "label" -> "mnist/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0"))

    val testDataIter = IO.MNISTIter(Map(
      "image" -> "mnist/t10k-images-idx3-ubyte",
      "label" -> "mnist/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0"))

    // define the neural network architecture
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> numHidden))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act1, "num_hidden" -> numOutputs))
    val mlp = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc2))

    // create and train the model
    val model = FeedForward.newBuilder(mlp)
      .setContext(Context.cpu()) // change to gpu if available
      .setTrainData(trainDataIter)
      .setEvalData(testDataIter)
      .setNumEpoch(numEpochs)
      .setOptimizer(new SGD(learningRate = learningRate))
      .setInitializer(new Xavier()) // random weight initialization
      .build()

    // evaluate model performance
    def accuracy(dataset: DataIter): Float = {
      dataset.reset()
      val predictions = model.predict(dataset).head
      // get predicted labels
      val predictedY = NDArray.argmax_channel(predictions)

      // get real labels
      dataset.reset()
      val labels = dataset.map(_.label(0).copy()).toVector
      val y = NDArray.concatenate(labels)
      require(y.shape == predictedY.shape)

      // calculate accuracy
      val numCorrect = (y.toArray zip predictedY.toArray).count {
        case (labelElem, predElem) => labelElem == predElem
      }
      numCorrect.toFloat / y.size
    }

    println(s"Train accuracy = ${accuracy(trainDataIter)}")
    println(s"Test accuracy = ${accuracy(testDataIter)}")
  }
}
