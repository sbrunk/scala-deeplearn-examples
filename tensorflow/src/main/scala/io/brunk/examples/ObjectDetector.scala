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

import java.io.{BufferedInputStream, File, FileInputStream}
import java.nio.ByteBuffer
import javax.swing.JFrame

import object_detection.protos.string_int_label_map.{StringIntLabelMap, StringIntLabelMapItem}
import org.bytedeco.javacpp.opencv_core.{FONT_HERSHEY_PLAIN, Mat, Point, Scalar}
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.{COLOR_BGR2RGB, cvtColor, putText, rectangle}
import org.bytedeco.javacv.{CanvasFrame, OpenCVFrameConverter, OpenCVFrameGrabber}
import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor, UINT8}
import org.tensorflow.framework.GraphDef

import scala.collection.Iterator.continually
import scala.io.Source

case class DetectionOutput(boxes: Tensor, scores: Tensor, classes: Tensor, num: Tensor)

/**
  * This example shows how to run a pretrained TensorFlow object detection model i.e. one from
  * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  *
  * You have to download and extract the model you want to run first, like so:
  * $ mkdir models && cd models
  * $ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
  * $ tar xzf ssd_inception_v2_coco_2017_11_17.tar.gz
  *
  * @author Sören Brunk
  */
object ObjectDetector {

  def main(args: Array[String]): Unit = {

    def printUsageAndExit(): Unit = {
      Console.err.println(
        """
          |Usage: ObjectDetector image <file>|video <file>|camera <deviceno> [<modelpath>]
          |  <file>      path to an image/video file
          |  <deviceno>  camera device number (usually starts with 0)
          |  <modelpath> optional path to the object detection model to be used. Default: ssd_inception_v2_coco_2017_11_17
          |""".stripMargin.trim)
      sys.exit(2)
    }

    if (args.length < 2) printUsageAndExit()

    val modelDir = args.lift(2).getOrElse("ssd_inception_v2_coco_2017_11_17")
    // load a pretrained detection model as TensorFlow graph
    val graphDef = GraphDef.parseFrom(
      new BufferedInputStream(new FileInputStream(new File(new File("models", modelDir), "frozen_inference_graph.pb"))))
    val graph = Graph.fromGraphDef(graphDef)

    // create a session and add our pretrained graph to it
    val session = Session(graph)

    // load the protobuf label map containing the class number to string label mapping (from COCO)
    val labelMap: Map[Int, String] = {
      val pbText = Source.fromResource("mscoco_label_map.pbtxt").mkString
      val stringIntLabelMap = StringIntLabelMap.fromAscii(pbText)
      stringIntLabelMap.item.collect {
        case StringIntLabelMapItem(_, Some(id), Some(displayName)) => id -> displayName
      }.toMap
    }

    val inputType = args(0)
    inputType match {
      case "image" =>
        val image = imread(args(1))
        detectImage(image, graph, session, labelMap)
      case "video" =>
        val grabber = new OpenCVFrameGrabber(args(1))
        detectSequence(grabber, graph, session, labelMap)
      case "camera" =>
        val cameraDevice = Integer.parseInt(args(1))
        val grabber = new OpenCVFrameGrabber(cameraDevice)
        detectSequence(grabber, graph, session, labelMap)
      case _ => printUsageAndExit()
    }
  }

  // convert OpenCV tensor to TensorFlow tensor
  def matToTensor(image: Mat): Tensor = {
    val imageRGB = new Mat
    cvtColor(image, imageRGB, COLOR_BGR2RGB) // convert channels from OpenCV GBR to RGB
    val imgBuffer = imageRGB.createBuffer[ByteBuffer]
    val shape = Shape(1, image.size.height, image.size.width(), image.channels)
    Tensor.fromBuffer(UINT8, shape, imgBuffer.capacity, imgBuffer)
  }

  // run detector on a single image
  def detectImage(image: Mat, graph: Graph, session: Session, labelMap: Map[Int, String]): Unit = {
    val canvasFrame = new CanvasFrame("Object Detection")
    canvasFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE) // exit when the canvas frame is closed
    canvasFrame.setCanvasSize(image.size.width, image.size.height)
    val detectionOutput = detect(matToTensor(image), graph, session)
    drawBoundingBoxes(image, labelMap, detectionOutput)
    canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(image))
    canvasFrame.waitKey(0)
    canvasFrame.dispose()
  }

  // run detector on an image sequence
  def detectSequence(grabber: OpenCVFrameGrabber, graph: Graph, session: Session, labelMap: Map[Int, String]): Unit = {
    val canvasFrame = new CanvasFrame("Object Detection", CanvasFrame.getDefaultGamma / grabber.getGamma)
    canvasFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE) // exit when the canvas frame is closed
    grabber.start()
    for (frame <- continually(grabber.grab()).takeWhile(_ != null
      && (grabber.getLengthInFrames == 0 || grabber.getFrameNumber < grabber.getLengthInFrames))) {
      val converter = new OpenCVFrameConverter.ToMat()
      val image = converter.convert(frame)

      val detectionOutput = detect(matToTensor(image), graph, session)
      drawBoundingBoxes(image, labelMap, detectionOutput)

      if (canvasFrame.isVisible) { // show our frame in the preview
        canvasFrame.showImage(frame)
      }
    }
    canvasFrame.dispose()
    grabber.stop()
  }

  // run the object detection model on an image
  def detect(image: Tensor, graph: Graph, session: Session): DetectionOutput = {

    // retrieve the output placeholders
    val imagePlaceholder = graph.getOutputByName("image_tensor:0")
    val detectionBoxes = graph.getOutputByName("detection_boxes:0")
    val detectionScores = graph.getOutputByName("detection_scores:0")
    val detectionClasses = graph.getOutputByName("detection_classes:0")
    val numDetections = graph.getOutputByName("num_detections:0")

    // set image as input parameter
    val feeds = Map(imagePlaceholder -> image)

    // Run the detection model
    val Seq(boxes, scores, classes, num) =
      session.run(fetches = Seq(detectionBoxes, detectionScores, detectionClasses, numDetections), feeds = feeds)
    DetectionOutput(boxes, scores, classes, num)
  }

  // draw boxes with class and score around detected objects
  def drawBoundingBoxes(image: Mat, labelMap: Map[Int, String], detectionOutput: DetectionOutput): Unit = {
    for (i <- 0 until detectionOutput.boxes.shape.size(1)) {
      val score = detectionOutput.scores(0, i).scalar.asInstanceOf[Float]

      if (score > 0.5) {
        val box = detectionOutput.boxes(0, i).entriesIterator.map(_.asInstanceOf[Float]).toSeq
        // we have to scale the box coordinates to the image size
        val ymin = (box(0) * image.size().height()).toInt
        val xmin = (box(1) * image.size().width()).toInt
        val ymax = (box(2) * image.size().height()).toInt
        val xmax = (box(3) * image.size().width()).toInt
        val label = labelMap.getOrElse(detectionOutput.classes(0, i).scalar.asInstanceOf[Float].toInt, "unknown")

        // draw score value
        putText(image,
          f"$label%s ($score%1.2f)", // text
          new Point(xmin, ymin - 6), // text position
          FONT_HERSHEY_PLAIN, // font type
          1.5, // font scale
          new Scalar(0, 255, 0, 0), // text color
          2, // text thickness
          8, // line type
          false) // origin is at the top-left corner
        // draw bounding box
        rectangle(image,
          new Point(xmin, ymin), // upper left corner
          new Point(xmax, ymax), // lower right corner
          new Scalar(0, 255, 0, 0), // color
          2, // thickness
          0, // lineType
          0) // shift
      }
    }
  }
}
