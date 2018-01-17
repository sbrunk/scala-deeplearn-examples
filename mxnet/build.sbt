// *****************************************************************************
// Projects
// *****************************************************************************

lazy val mxnet =
  project
    .in(file("."))
    .enablePlugins(AutomateHeaderPlugin)
    .settings(settings)
    .settings(
      scalaVersion := "2.11.12", // MXNet is only available for Scala 2.11
      resolvers += Resolver.mavenLocal,
      libraryDependencies ++= Seq(
        library.logbackClassic,
        library.mxnetFull
      )
    )

// *****************************************************************************
// Library dependencies
// *****************************************************************************

lazy val library =
  new {
    object Version {
      val logbackClassic = "1.2.3"
      val mxnet = "1.0.0-SNAPSHOT"
    }
    val logbackClassic = "ch.qos.logback" % "logback-classic" % Version.logbackClassic
    // change to "mxnet-full_2.10-linux-x86_64-cpu" or "mxnet-full_2.10-linux-x86_64-gpu" depending on your os/gpu
    val mxnetFull = "ml.dmlc.mxnet" % "mxnet-full_2.11-osx-x86_64-cpu" % Version.mxnet
  }

// *****************************************************************************
// Settings
// *****************************************************************************

lazy val settings =
  Seq(
    scalaVersion := "2.12.4",
    organization := "io.brunk",
    organizationName := "Sören Brunk",
    startYear := Some(2017),
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0")),
    scalacOptions ++= Seq(
      "-unchecked",
      "-deprecation",
      "-language:_",
      "-target:jvm-1.8",
      "-encoding", "UTF-8"
    ),
    unmanagedSourceDirectories.in(Compile) := Seq(scalaSource.in(Compile).value),
    unmanagedSourceDirectories.in(Test) := Seq(scalaSource.in(Test).value)
)
