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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.nd4j.linalg.learning.config.Adam
import io.brunk.examples.ImageReader._
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation.{RELU, SOFTMAX}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory


object SimpleCNN {

  private val log = LoggerFactory.getLogger(getClass)
  val seed = 1

  def main(args: Array[String]): Unit = {

    val dataDir = args.head

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .updater(new Adam)
      .list()
      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .nIn(channels)
        .nOut(32)
        .activation(RELU)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build())
      .layer(2, new ConvolutionLayer.Builder(3, 3)
        .nOut(64)
        .activation(RELU)
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build())
      .layer(4, new ConvolutionLayer.Builder(3, 3)
        .nOut(128)
        .activation(RELU)
        .build())
      .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build())
      .layer(6, new ConvolutionLayer.Builder(3, 3)
        .nOut(128)
        .activation(RELU)
        .build())
      .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build())
      .layer(8, new DenseLayer.Builder()
        .nOut(512)
        .activation(RELU)
        .dropOut(new Dropout(0.5))
        .build())
      .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(2)
        .activation(SOFTMAX)
        .build())
      .setInputType(InputType.convolutional(150, 150, 3))
      .backprop(true).pretrain(false).build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(10))
    log.debug("Total num of params: {}", model.numParams)

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

    val (trainIter, testIter) = createImageIterator(dataDir)

    model.fit(trainIter)
    val eval = model.evaluate(testIter)
    log.info(eval.stats)
  }
}
