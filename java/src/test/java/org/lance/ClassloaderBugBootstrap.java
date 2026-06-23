/*
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
package org.lance;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

/**
 * Standalone bootstrap for the forked-JVM classloader bug test.
 *
 * <p>This class has <b>zero</b> lance imports. It lives on the forked JVM's system classpath
 * ({@code target/test-classes/} only). All lance classes are loaded through an isolated {@link
 * URLClassLoader} whose parent is {@code null} (bootstrap classloader), so the JVM's system
 * classloader cannot see them.
 *
 * <p>Usage: {@code java -cp target/test-classes org.lance.ClassloaderBugBootstrap <full-classpath>
 * <temp-dir>}
 *
 * <p>Exit code 0 + "SUCCESS" on stdout = pass; non-zero = failure.
 */
public class ClassloaderBugBootstrap {

  public static void main(String[] args) {
    if (args.length < 2) {
      System.err.println("Usage: ClassloaderBugBootstrap <classpath> <tempDir>");
      System.exit(2);
    }

    String classpath = args[0];
    String tempDir = args[1];

    try {
      // Build URL array from the full classpath
      String[] entries = classpath.split(File.pathSeparator);
      URL[] urls = new URL[entries.length];
      for (int i = 0; i < entries.length; i++) {
        urls[i] = new File(entries[i]).toURI().toURL();
      }

      // Parent is null => only bootstrap classloader is the parent.
      // The system classloader (which only has target/test-classes/) is bypassed.
      URLClassLoader isolatedCl = new URLClassLoader(urls, null);

      // Set as context classloader for consistency
      Thread.currentThread().setContextClassLoader(isolatedCl);

      // Load and invoke the helper class through the isolated classloader
      Class<?> helperClass = Class.forName("org.lance.ClassloaderBugHelper", true, isolatedCl);
      Method runMethod = helperClass.getMethod("run", String.class);
      runMethod.invoke(null, tempDir);

      // Force exit: the JNI dispatcher thread is non-daemon and would keep the JVM alive
      System.exit(0);

    } catch (Throwable t) {
      System.err.println("ClassloaderBugBootstrap FAILED:");
      t.printStackTrace(System.err);
      System.exit(1);
    }
  }
}
