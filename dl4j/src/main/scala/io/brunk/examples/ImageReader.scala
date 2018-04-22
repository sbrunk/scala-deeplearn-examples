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

import java.io.{File, FileFilter}
import java.lang.Math.toIntExact

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler

import scala.collection.JavaConverters._


object ImageReader {

  val channels = 3
  val height = 150
  val width = 150

  val batchSize = 50
  val numClasses = 2
  val epochs = 100
  val splitTrainTest = 0.8

  val random = new java.util.Random()

  def createImageIterator(path: String): (MultipleEpochsIterator, DataSetIterator) = {
    val baseDir = new File(path)
    val labelGenerator = new ParentPathLabelGenerator
    val fileSplit = new FileSplit(baseDir, BaseImageLoader.ALLOWED_FORMATS, random)

    val numExamples = toIntExact(fileSplit.length)
    val numLabels = fileSplit.getRootDir.listFiles(new FileFilter {
      override def accept(pathname: File): Boolean = pathname.isDirectory
    }).length

    val pathFilter = new BalancedPathFilter(random, labelGenerator, numExamples, numLabels, batchSize)

    //val inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest)
    val inputSplit = fileSplit.sample(pathFilter, 70, 30)

    val trainData = inputSplit(0)
    val validationData = inputSplit(1)

    val recordReader = new ImageRecordReader(height, width, channels, labelGenerator)
    val scaler = new ImagePreProcessingScaler(0, 1)

    recordReader.initialize(trainData, null)
    val dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    val trainIter = new MultipleEpochsIterator(epochs, dataIter)

    val valRecordReader = new ImageRecordReader(height, width, channels, labelGenerator)
    valRecordReader.initialize(validationData, null)
    val validationIter = new RecordReaderDataSetIterator(valRecordReader, batchSize, 1, numClasses)
    scaler.fit(validationIter)
    validationIter.setPreProcessor(scaler)

    (trainIter, validationIter)
  }

}
