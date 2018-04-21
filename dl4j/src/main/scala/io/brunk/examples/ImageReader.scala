package io.brunk.examples

import java.io.File

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler


object ImageReader {

  val channels = 3
  val height = 150
  val width = 150
  val random = new java.util.Random()

  val batchSize = 10
  val numClasses = 2

  def createImageIterator(path: String): (DataSetIterator, DataSetIterator) = {
    val baseDir = new File(path)
    val labelGenerator = new ParentPathLabelGenerator
    val inputSplit = new FileSplit(baseDir, BaseImageLoader.ALLOWED_FORMATS, random)
    val trainTest = inputSplit.sample(null, 60, 20, 20)
    val trainData = trainTest(0)

    val trainRR = new ImageRecordReader(height, width, channels, labelGenerator)
    trainRR.initialize(trainData)
    val trainIter = new RecordReaderDataSetIterator(trainRR, batchSize,1,  numClasses)
    val scaler = new ImagePreProcessingScaler(0, 1)
    scaler.fit(trainIter)
    trainIter.setPreProcessor(scaler)

    val validationData = trainTest(1)
    val  validationRR = new ImageRecordReader(height, width, channels, labelGenerator)
    validationRR.initialize(validationData)
    val validationIter = new RecordReaderDataSetIterator(validationRR, batchSize,1,  numClasses)
    validationIter.setPreProcessor(scaler)

    (trainIter, validationIter)
  }

}
