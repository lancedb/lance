scalaVersion := "2.12.17" // align with Spark

name         := "jlance"
organization := "eto.ai"
version      := "1.0"
val arrowVersion = "11.0.0"
libraryDependencies ++= Seq(
  "org.apache.arrow" % "arrow-c-data"       % arrowVersion,
  "org.apache.arrow" % "arrow-memory-netty" % arrowVersion,
// remove it after scala 2.13
  "com.github.bigwheel" %% "util-backports" % "2.1"
)
