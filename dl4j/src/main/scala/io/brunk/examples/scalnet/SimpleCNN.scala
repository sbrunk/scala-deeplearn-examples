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

import io.brunk.examples.ImageReader
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.scalnet.models.NeuralNet
import io.brunk.examples.ImageReader._
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.nd4j.linalg.activations.Activation._
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object SimpleCNN {


  def main(args: Array[String]): Unit = {

    val dataDir = args.head

    val seed = 1

    val model = NeuralNet(inputType = InputType.convolutional(height, width, channels), rngSeed = seed)

    model.add(Convolution2D(32, List(3, 3), channels, activation = RELU))
    model.add(MaxPooling2D(List(2, 2)))

    model.add(Convolution2D(64, List(3, 3), activation = RELU))
    model.add(MaxPooling2D(List(2, 2)))

    model.add(Convolution2D(128, List(3, 3), activation = RELU))
    model.add(MaxPooling2D(List(2, 2)))

    model.add(Convolution2D(128, List(3, 3), activation = RELU))
    model.add(MaxPooling2D(List(2, 2)))

    model.add(Dense(512, activation = RELU, dropOut = 0.5))
    model.add(Dense(2, activation = SOFTMAX))

    model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD, updater = Updater.ADAM)

    val (trainIter, testIter) = createImageIterator(dataDir)

    model.fit(trainIter, 30, List(new ScoreIterationListener(10)))
  }
}
