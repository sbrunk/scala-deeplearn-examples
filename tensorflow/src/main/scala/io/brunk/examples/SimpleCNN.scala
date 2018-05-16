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

import java.nio.ByteBuffer
import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.{DataType, _}
import org.platanios.tensorflow.api.ops.NN.ValidConvPadding
import org.slf4j.LoggerFactory
import better.files._
import javax.swing.JFrame
import org.bytedeco.javacpp.opencv_core.{Mat, Point, Scalar}
import org.bytedeco.javacpp.opencv_imgproc.{COLOR_BGR2RGB, cvtColor, resize}
import org.bytedeco.javacv._

import scala.util.Random.shuffle
import org.platanios.tensorflow.api.ops.io.data.{Dataset, TensorSlicesDataset}
import org.platanios.tensorflow.api.ops.variables.GlorotUniformInitializer
import org.bytedeco.javacpp.opencv_imgproc.{COLOR_BGR2RGB, cvtColor, putText, rectangle}
import org.bytedeco.javacpp.opencv_core.{repeat => _, _}
import org.bytedeco.javacpp.opencv_imgcodecs.imread

import scala.collection.Iterator.continually

/**
  * CNN for image classification example
  *
  * @author Sören Brunk
  */
object SimpleCNN {

  private[this] val logger = Logger(LoggerFactory.getLogger(getClass))

  def main(args: Array[String]): Unit = {

    val seed = 42
    val batchSize    = 64

    val dataDir = File(args(0))
    val mode = args(1) // train or infer

    // define the neural network architecture
    val input = tf.learn.Input(UINT8, Shape(-1, 250, 250, 3)) // type and shape of images
    val trainInput = tf.learn.Input(UINT8, Shape(-1)) // type and shape of labels

    val modelIndex = args(2).toInt
    val layers = SimpleCNNModels.models(modelIndex)

    val labelMap = Seq("not_scala", "scala")

    val trainInputLayer = tf.learn.Cast("TrainInput/Cast", INT64) // cast labels to long

    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
      tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.Adam(0.001f)

    val model = tf.learn.Model.supervised(input, layers, trainInput, trainInputLayer, loss, optimizer)

    val summariesDir = Paths.get(s"temp/logo-classifier-v$modelIndex")
    val accMetric = tf.metrics.MapMetric(
      (v: (Output, Output)) => (v._1.argmax(-1), v._2), tf.metrics.Accuracy())

    mode match {
      case "train" => train()
      case "infer" => infer()
    }

    // train the model
    def train(): Unit = {
      val trainDir = dataDir / "train"
      val testDir = dataDir / "validation"
      val imgClassDirs = trainDir.list.filter(_.isDirectory).toVector.sortBy(_.name)
      val numClasses = imgClassDirs.size
      logger.info("Number of classes {}", numClasses)

      val numericLabelForClass = imgClassDirs.map(_.name).zipWithIndex.toMap
      logger.info("classes {}", numericLabelForClass)

      def filenamesWithLabels(dir: File): (Tensor, Tensor) = {
        val (filenames, labels) = (for {
          dir <- dir.children.filter(_.isDirectory)
          filename <- dir.glob("*.jpg").map(_.pathAsString)
        } yield (filename, numericLabelForClass(dir.name))).toVector.unzip
        (Tensor(filenames).squeeze(Seq(0)), Tensor(UINT8, labels).squeeze(Seq(0))
        )
      }

      def readImage(filename: Output): Output = {
        val rawImage = tf.data.readFile(filename)
        val image = tf.image.decodeJpeg(rawImage, numChannels = 3)
        tf.image.resizeBilinear(image.expandDims(axis = 0), Seq(250, 250)).squeeze(Seq(0)).cast(UINT8)
      }

      val trainData: Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] =
        tf.data.TensorSlicesDataset(filenamesWithLabels(trainDir))
          .shuffle(bufferSize = 30000, Some(seed))
          .map({ case (filename, label) => (readImage(filename), label)}, numParallelCalls = 16)
          .cache("")
          .repeat()
          .batch(batchSize)
          .prefetch(100)

      val evalTrainData: Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] =
        tf.data.TensorSlicesDataset(filenamesWithLabels(trainDir))
          .shuffle(bufferSize = 30000, Some(seed))
          .map({ case (filename, label) => (readImage(filename), label)}, numParallelCalls = 16)
          .take(2000)
          .cache("")
          .batch(128)
          .prefetch(100)

      val evalTestData: Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] =
        tf.data.TensorSlicesDataset(filenamesWithLabels(testDir))
          .shuffle(bufferSize = 2000, Some(seed))
          .map({ case (filename, label) => (readImage(filename), label)}, numParallelCalls = 16)
          .cache("")
          .batch(128)
          .prefetch(100)

      val estimator = tf.learn.InMemoryEstimator(
        model,
        tf.learn.Configuration(Some(summariesDir)),
        tf.learn.StopCriteria(maxSteps = Some(10000)),
        Set(
          tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
          tf.learn.Evaluator(
            log = true, datasets = Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
            metrics = Seq(accMetric), trigger = tf.learn.StepHookTrigger(100), name = "Evaluator",
            summaryDir = summariesDir),
          tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
          tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
          tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
        tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))

      estimator.train(() => trainData)
    }

    def infer(): Unit = {

      val estimator = tf.learn.InMemoryEstimator(model, tf.learn.Configuration(Some(summariesDir)))

      val inputType = args(3)
      val input = args(4)
      inputType match {
        case "image" =>
          val image = imread(input)
          detectImage(image)
        case "video" =>
          val grabber = new FFmpegFrameGrabber(input)
          detectSequence(grabber)
        case "camera" =>
          val cameraDevice = Integer.parseInt(input)
          val grabber = new OpenCVFrameGrabber(cameraDevice)
          detectSequence(grabber)
        case _ => sys.exit(1)
      }

      // convert OpenCV tensor to TensorFlow tensor
      def matToTensor(image: Mat): Tensor = {
        val imageRGB = new Mat
        cvtColor(image, imageRGB, COLOR_BGR2RGB) // convert channels from OpenCV GBR to RGB
        val imgBuffer = imageRGB.createBuffer[ByteBuffer]
        val shape = Shape(1, image.size.height, image.size.width(), image.channels)
        Tensor.fromBuffer(UINT8, shape, imgBuffer.capacity, imgBuffer)
      }

      def drawLabel(image: Mat, label: String): Unit =
        putText(image,
          label, // text
          new Point(50, 50), // text position
          FONT_HERSHEY_PLAIN, // font type
          2.6, // font scale
          new Scalar(0, 0, 100, 0), // text color
          4, // text thickness
          LINE_AA, // line type
          false) // origin is at the top-left corner

      // run detector on a single image
      def detectImage(image: Mat): Unit = {
        val canvasFrame = new CanvasFrame("Logo Classifier")
        canvasFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE) // exit when the canvas frame is closed
        canvasFrame.setCanvasSize(image.size.width, image.size.height)

        val imageTensor = matToTensor(image)
        val s = Session()
        val resized = s.run(fetches = tf.image.resizeBilinear(imageTensor, Seq(250, 250)).cast(UINT8))

        val result: Tensor = estimator.infer(() => resized)

        logger.info("Result {}", result.summarize(flattened = true))
        val probabilities = result.softmax().entriesIterator.map(_.asInstanceOf[Float]).toVector
        logger.info("Probabilities {}", probabilities)
        val label = result.argmax(-1).scalar.asInstanceOf[Long].toInt
        logger.info("Label {}", label.summarize(flattened = true))

        drawLabel(image,
          s"Class: $label (${labelMap(label)}) " +
          s"Probability(${probabilities(label)})")

        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(image))
        canvasFrame.waitKey(0)
        canvasFrame.dispose()
      }

      // run detector on an image sequence
      def detectSequence(grabber: FrameGrabber): Unit = {
        val canvasFrame = new CanvasFrame("Logo Classifier", CanvasFrame.getDefaultGamma / grabber.getGamma)
        canvasFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE) // exit when the canvas frame is closed
        val converter = new OpenCVFrameConverter.ToMat()
        grabber.start()
        for (frame <- continually(grabber.grab()).takeWhile(_ != null
          && (grabber.getLengthInFrames == 0 || grabber.getFrameNumber < grabber.getLengthInFrames))) {
          val image = converter.convert(frame)
          if (image != null) { // sometimes the first few frames are empty so we ignore them

            val imageTensor = matToTensor(image)
            val s = Session()
            val resized = s.run(fetches = tf.image.resizeBilinear(imageTensor, Seq(250, 250)).cast(UINT8))

            val result: Tensor = estimator.infer(() => resized)

            logger.info("Result {}", result.summarize(flattened = true))
            val probabilities = result.softmax().entriesIterator.map(_.asInstanceOf[Float]).toVector
            logger.info("Probabilities {}", probabilities)
            val label = result.argmax(-1).scalar.asInstanceOf[Long].toInt
            logger.info("Label {}", label.summarize(flattened = true))

            drawLabel(image,
              s"Class: $label (${labelMap(label)}) " +
                s"Probability: ${probabilities(label)}")

            if (canvasFrame.isVisible) { // show our frame in the preview
              canvasFrame.showImage(frame)
            }
          }
        }
        canvasFrame.dispose()
        grabber.stop()
      }
    }


    //def accuracy(images: Tensor, labels: Tensor): Float = {
    //  val predictions = estimator.infer(() => images)
    //  predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
    //}

    // evaluate model performance
    //logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    //logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")

  }
}
