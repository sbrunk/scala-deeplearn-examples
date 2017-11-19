// *****************************************************************************
// Projects
// *****************************************************************************

lazy val `scala-deeplearn-examples` =
  project
    .in(file("."))
    .enablePlugins(AutomateHeaderPlugin)
    .settings(settings)
    .settings(
      libraryDependencies ++= Seq(
        library.scalaCheck % Test,
        library.scalaTest  % Test
      )
    )


lazy val dl4j =
  project
    .in(file("dl4j"))
    .enablePlugins(AutomateHeaderPlugin)
    .settings(settings)
    .settings(
      scalaVersion := "2.11.12", // ScalNet and ND4S is only available for Scala 2.11
      libraryDependencies ++= Seq(
        library.dl4j,
        library.logbackClassic,
        library.nd4jNativePlatform,
        library.scalNet
      )
    )

// *****************************************************************************
// Library dependencies
// *****************************************************************************

lazy val library =
  new {
    object Version {
      val dl4j = "0.9.1"
      val logbackClassic = "1.2.3"
      val scalaCheck = "1.13.5"
      val scalaTest  = "3.0.4"
    }
    val dl4j = "org.deeplearning4j" % "deeplearning4j-core" % Version.dl4j
    val logbackClassic = "ch.qos.logback" % "logback-classic" % Version.logbackClassic
    val nd4jNativePlatform = "org.nd4j" % "nd4j-native-platform" % Version.dl4j
    val scalaCheck = "org.scalacheck" %% "scalacheck" % Version.scalaCheck
    val scalaTest  = "org.scalatest"  %% "scalatest"  % Version.scalaTest
    val scalNet = "org.deeplearning4j" %% "scalnet" % Version.dl4j
  }

// *****************************************************************************
// Settings
// *****************************************************************************

lazy val settings =
  commonSettings ++
  scalafmtSettings

lazy val commonSettings =
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
    unmanagedSourceDirectories.in(Test) := Seq(scalaSource.in(Test).value),
    wartremoverWarnings in (Compile, compile) ++= Warts.unsafe,
    resolvers ++= Seq(
      Resolver.sonatypeRepo("snapshots")
    )
)

lazy val scalafmtSettings =
  Seq(
    scalafmtOnCompile := true,
    scalafmtOnCompile.in(Sbt) := false,
    scalafmtVersion := "1.3.0"
  )
