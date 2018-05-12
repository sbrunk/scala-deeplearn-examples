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

package io.brunk

import better.files.File

import scala.util.Random.shuffle

/** Script that splits an image dataset into train/validation/test set
  *
  * Expects the following structure per class: <class name>/<images.jpg>
  * Outputs each subset into a subdir for training, validation and testset
  *
  * usage: DatasetSplitter <input dir> <output dir> [<trainsize> <validationsize> <testsize>]
  * Sizes in %
  */
object DatasetSplitter {
  val datasetSplitNames = Seq("train", "validation", "test")

  def main(args: Array[String]): Unit = {
    val inputDir = File(args(0))
    val outputDir = File(args(1))
    val splitSizes = args.drop(2).map(_.toFloat).toSeq

    val imgClassDirs = inputDir.list.filter(_.isDirectory).toVector.sortBy(_.name)

    // in case of different numbers of samples per class, use the smallest one
    val numSamplesPerClass = imgClassDirs.map(_.glob("*.jpg").size).min
    println(s"Number of samples per class (balanced): $numSamplesPerClass")

    val samplesPerClass =  {
      imgClassDirs.flatMap { imgClassDir =>
        shuffle(imgClassDir.children.toVector)
          .take(numSamplesPerClass)       // balance samples to have the same number for each class
          .map((imgClassDir.name, _))
      }.groupBy(_._1).mapValues(_.map(_._2))
    }

    val numSamples = samplesPerClass.map(_._2.size).sum
    println(s"Number of samples (balanced): $numSamples")
    val splitSizesAbsolute = splitSizes.map(_ / 100.0).map(_ * numSamplesPerClass).map(_.toInt)
    println(s"Number of samples per split: ${datasetSplitNames.zip(splitSizesAbsolute).mkString(" ")}")
    val splitIndices = splitSizesAbsolute
      .map(_ -1)
      .scanLeft(-1 to -1)((prev, current) => prev.last + 1 to (prev.last + current + 1)).tail // TODO cleaner solution
    println(splitIndices)

    val datasetNamesWithIndices = datasetSplitNames.zip(splitIndices)

    val datasetIndices = (for {
      (name, indices) <- datasetNamesWithIndices
      index <- indices
    } yield (index, name))
      .sortBy(_._1)
      .map(_._2)

    // create directories
    for {
      dataset <- datasetSplitNames
      imgClassDir <- imgClassDirs
      imgClass = imgClassDir.name
    } {
      (outputDir/dataset/imgClass).createDirectories()
    }
    // write into train, validation and test folders
    for {
      (imgClass, samples) <- samplesPerClass
      (filename, dataset) <- samples.zip(datasetIndices)
    } {
      filename.copyTo(outputDir/dataset/imgClass/filename.name)
    }

  }
}
