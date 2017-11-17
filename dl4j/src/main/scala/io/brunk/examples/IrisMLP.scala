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

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ DenseLayer, OutputLayer }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Based on
  * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java
  */
object IrisMLP {
  val numLinesToSkip = 1

  val batchSize  = 200
  val labelIndex = 4
  val numLabels  = 3

  val seed       = 1
  val iterations = 1000
  val numInputs  = 4

  def readData(): SplitTestAndTrain = {
    val recordReader = new CSVRecordReader(numLinesToSkip, ',')
    recordReader.initialize(new FileSplit(new ClassPathResource("iris.csv").getFile))
    val iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numLabels)
    val dataSet  = iterator.next() // read all data in a single batch
    dataSet.shuffle(1)
    val testAndTrain = dataSet.splitTestAndTrain(0.70)
    val train        = testAndTrain.getTrain
    val test         = testAndTrain.getTest

    val normalizer = new NormalizerStandardize
    normalizer.fit(train)
    normalizer.transform(train) // normalize training data
    normalizer.transform(test)  // normalize test data
    testAndTrain
  }

  def main(args: Array[String]): Unit = {

    val testAndTrain = readData()

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(Activation.RELU)
      .weightInit(WeightInit.XAVIER)
      .learningRate(0.1)
      .regularization(true)
      .l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(4).build())
      .layer(1, new DenseLayer.Builder().nIn(4).nOut(3).build())
      .layer(2,
             new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
               .activation(Activation.SOFTMAX)
               .nIn(3)
               .nOut(numLabels)
               .build())
      .backprop(true)
      .pretrain(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100)) // print out scores every 100 iterations
    model.fit(testAndTrain.getTrain)

    val eval   = new Evaluation(numLabels)
    val output = model.output(testAndTrain.getTest.getFeatureMatrix)
    eval.eval(testAndTrain.getTest.getLabels, output)
    println(eval.stats)
  }
}
