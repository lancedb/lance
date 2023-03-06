scalaVersion := "2.12.17" // align with Spark

name         := "jlance"
organization := "eto.ai"
version      := "1.0"
val arrowVersion = "11.0.0"
libraryDependencies ++= Seq(
  "org.apache.arrow"     % "arrow-c-data"       % arrowVersion,
  "org.apache.arrow"     % "arrow-memory-netty" % arrowVersion,
  "org.apache.arrow"     % "arrow-dataset" % arrowVersion,
  "ch.qos.logback" % "logback-classic" % "1.2.11",
// remove it after scala 2.13
  "com.github.bigwheel" %% "util-backports"     % "2.1"
)
run / fork := true
run / javaOptions += "-Djava.library.path=../target/debug/"

ThisBuild / assemblyMergeStrategy := {
  case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
  case "application.conf"                            => MergeStrategy.concat
  case "unwanted.txt"                                => MergeStrategy.discard
  case "arrow-git.properties"                        => MergeStrategy.first
  case PathList("META-INF", xs @ _*)                 => MergeStrategy.first
  case x                                             =>
    val oldStrategy = (ThisBuild / assemblyMergeStrategy).value
    oldStrategy(x)
}
