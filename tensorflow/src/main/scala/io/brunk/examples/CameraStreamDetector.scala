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

import java.io.{BufferedInputStream, FileInputStream}
import java.nio.ByteBuffer

import object_detection.protos.string_int_label_map.{StringIntLabelMap, StringIntLabelMapItem}
import org.bytedeco.javacpp.opencv_core.{FONT_HERSHEY_PLAIN, Mat, Point, Scalar}
import org.bytedeco.javacpp.opencv_imgproc.{putText, rectangle}
import org.bytedeco.javacv.{Frame, OpenCVFrameConverter, OpenCVFrameGrabber}
import org.bytedeco.javacv.CanvasFrame

import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor, UINT8}
import org.tensorflow.framework.GraphDef

import scala.collection.Iterator.continually
import scala.io.Source


object CameraStreamDetector {
  def main(args: Array[String]): Unit = {
    val captureWidth = 1280
    val captureHeight = 720

    val grabber = new OpenCVFrameGrabber(0)
    grabber.setImageWidth(captureWidth)
    grabber.setImageHeight(captureHeight)
    grabber.start()

    // read an image using OpenCV and convert to tensor
    def matToTensor(image: Mat) = {
      val shape = Shape(1, image.size().height(), image.size().width(), image.channels())
      val imgBuffer = image.createBuffer[ByteBuffer]
      val imgTensorBGR = Tensor.fromBuffer(UINT8, shape, imgBuffer.capacity, imgBuffer)
      imgTensorBGR.reverse(Seq(-1)) // convert channels from OpenCV GBR to RGB
    }

    val labelMap: Map[Int, String] = {
      val pbtext = Source.fromResource("mscoco_label_map.pbtxt").mkString
      val stringIntLabelMap = StringIntLabelMap.fromAscii(pbtext)
      stringIntLabelMap.item.collect {
        case StringIntLabelMapItem(_, Some(id), Some(displayName)) => id -> displayName
      }.toMap
    }

    // load a pretrained detection model as TensorFlow graph
    val graphDef = GraphDef.parseFrom(
      new BufferedInputStream(
        new FileInputStream("models/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb")))
    val graph = Graph.fromGraphDef(graphDef)

    // retrieve the output placeholders
    val imagePlaceholder = graph.getOutputByName("image_tensor:0")
    val detectionBoxes = graph.getOutputByName("detection_boxes:0")
    val detectionScores = graph.getOutputByName("detection_scores:0")
    val detectionClasses = graph.getOutputByName("detection_classes:0")
    val numDetections = graph.getOutputByName("num_detections:0")

    // create a session and add our pretrained graph to it
    val sess = Session(graph)
    val cFrame = new CanvasFrame("Capture Preview", CanvasFrame.getDefaultGamma / grabber.getGamma)

    // While we are capturing...
    for (frame <- continually(grabber.grab()).takeWhile(_ != null)) {
      val converter = new OpenCVFrameConverter.ToMat()
      val image = converter.convert(frame)

      // set image as input parameter
      val feeds = Map(imagePlaceholder -> matToTensor(image))

      // Run the detection model
      val Seq(boxes, scores, classes, num) =
        sess.run(fetches = Seq(detectionBoxes, detectionScores, detectionClasses, numDetections), feeds = feeds)

      // Draw boxes with class and score around detected objects
      for (i <- 0 until boxes.shape.size(1)) {
        val score = scores(0, i).scalar.asInstanceOf[Float]

        if (score > 0.5) {
          val box = boxes(0, i).entriesIterator.map(_.asInstanceOf[Float]).toSeq
          // We have to scale the box coordinates to the image size
          val ymin = (box(0) * image.size().height()).toInt
          val xmin = (box(1) * image.size().width()).toInt
          val ymax = (box(2) * image.size().height()).toInt
          val xmax = (box(3) * image.size().width()).toInt
          val label = labelMap.getOrElse(classes(0,i).scalar.asInstanceOf[Float].toInt, "unknown")

          putText(image,
            f"$label%s ($score%1.2f)", // text
            new Point(xmin, ymin - 6), // text position
            FONT_HERSHEY_PLAIN, // font type
            1.5, // font scale
            new Scalar(0, 255, 0, 0), // text color
            2, // text thickness
            8, // line type
            false) // origin is at the top-left corner
          rectangle(image,
            new Point(xmin, ymin), // upper left corner
            new Point(xmax, ymax), // lower right corner
            new Scalar(0, 255, 0, 0), // color
            2, // thickness
            0, // lineType
            0) // shift
        }
      }
      if (cFrame.isVisible) { // Show our frame in the preview
        cFrame.showImage(frame)
      }
    }
    cFrame.dispose()
    grabber.stop()
  }
}
