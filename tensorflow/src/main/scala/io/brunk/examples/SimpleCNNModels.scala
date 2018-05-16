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

import better.files._
import com.typesafe.scalalogging.Logger
import javax.swing.JFrame
import org.bytedeco.javacpp.opencv_core.{Mat, Point, Scalar, repeat => _, _}
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.{COLOR_BGR2RGB, cvtColor, putText, resize}
import org.bytedeco.javacv._
import org.platanios.tensorflow.api.ops.NN.ValidConvPadding
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.{DataType, _}
import org.platanios.tensorflow.api.tf.learn._

import org.slf4j.LoggerFactory

import scala.collection.Iterator.continually

/**
  * CNN for image classification example
  *
  * @author Sören Brunk
  */
object SimpleCNNModels {

  lazy val models = Seq(v0, v1, v2, v3, v4, v5, v6, v7)

  val v0 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Flatten("Layer_1/Flatten") >>
      tf.learn.Linear("Layer_1/Linear", units = 64) >>
      tf.learn.ReLU("Layer_1/ReLU", 0.01f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v1 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Flatten("Layer_1/Flatten") >>
      tf.learn.Linear("Layer_1/Linear", units = 128) >>
      tf.learn.ReLU("Layer_1/ReLU", 0.01f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v2 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Flatten("Layer_1/Flatten") >>
      tf.learn.Linear("Layer_1/Linear", units = 512) >>
      tf.learn.ReLU("Layer_1/ReLU") >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v3 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Flatten("Layer_1/Flatten") >>
      tf.learn.Linear("Layer_1/Linear", units = 512) >>
      tf.learn.ReLU("Layer_1/ReLU", 0.01f) >>
      tf.learn.Dropout("Layer_1/Dropout", keepProbability = 0.5f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v4 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Layer_1/Conv2D", Shape(3, 3, 3, 32), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_1/Bias") >>
      tf.learn.ReLU("Layer_1/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_1/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Flatten("Layer_2/Flatten") >>
      tf.learn.Linear("Layer_2/Linear", units = 512) >>
      tf.learn.ReLU("Layer_2/ReLU", 0.01f) >>
      tf.learn.Dropout("Layer_2/Dropout", keepProbability = 0.5f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v5 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Layer_1/Conv2D", Shape(3, 3, 3, 32), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_1/Bias") >>
      tf.learn.ReLU("Layer_1/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_1/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Conv2D("Layer_2/Conv2D", Shape(3, 3, 32, 64), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_2/Bias") >>
      tf.learn.ReLU("Layer_2/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_2/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Conv2D("Layer_3/Conv2D", Shape(3, 3, 64, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_3/Bias") >>
      tf.learn.ReLU("Layer_3/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_3/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Conv2D("Layer_4/Conv2D", Shape(3, 3, 128, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_4/Bias") >>
      tf.learn.ReLU("Layer_4/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_4/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Flatten("Layer_5/Flatten") >>
      tf.learn.Linear("Layer_5/Linear", units = 512) >>
      tf.learn.ReLU("Layer_5/ReLU", 0.01f) >>
      tf.learn.Dropout("Layer_3/Dropout", keepProbability = 0.5f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v6 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Layer_1/Conv2D", Shape(3, 3, 3, 32), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_1/Bias") >>
      tf.learn.ReLU("Layer_1/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_1/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Dropout("Layer_1/Dropout", keepProbability = 0.8f) >>
      tf.learn.Conv2D("Layer_2/Conv2D", Shape(3, 3, 32, 64), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_2/Bias") >>
      tf.learn.ReLU("Layer_2/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_2/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Dropout("Layer_2/Dropout", keepProbability = 0.8f) >>
      tf.learn.Conv2D("Layer_3/Conv2D", Shape(3, 3, 64, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_3/Bias") >>
      tf.learn.ReLU("Layer_3/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_3/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Dropout("Layer_3/Dropout", keepProbability = 0.8f) >>
      tf.learn.Conv2D("Layer_4/Conv2D", Shape(3, 3, 128, 128), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_4/Bias") >>
      tf.learn.ReLU("Layer_4/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_4/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Dropout("Layer_4/Dropout", keepProbability = 0.8f) >>
      tf.learn.Flatten("Layer_5/Flatten") >>
      tf.learn.Linear("Layer_5/Linear", units = 512) >>
      tf.learn.ReLU("Layer_5/ReLU", 0.01f) >>
      tf.learn.Dropout("Layer_5/Dropout", keepProbability = 0.8f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

  val v7 =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Layer_1/Conv2D", Shape(3, 3, 3, 32), stride1 = 1, stride2 = 1, padding = ValidConvPadding) >>
      tf.learn.AddBias("Layer_1/Bias") >>
      tf.learn.ReLU("Layer_1/ReLU", alpha = 0.1f) >>
      tf.learn.MaxPool("Layer_1/MaxPool", windowSize = Seq(1, 2, 2, 1), stride1 = 2, stride2 = 2, padding = ValidConvPadding) >>
      tf.learn.Flatten("Layer_2/Flatten") >>
      tf.learn.Linear("Layer_2/Linear", units = 512) >>
      tf.learn.ReLU("Layer_2/ReLU", 0.01f) >>
      tf.learn.Linear("OutputLayer/Linear", 2)

}
