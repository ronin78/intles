(defproject intles "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "1.0.0-beta"]
                 [org.nd4j/nd4j-api "1.0.0-beta"]
                 [org.nd4j/nd4j-native-platform "1.0.0-beta"]
                 [org.slf4j/slf4j-api "1.8.0-beta2"]
                 [org.slf4j/slf4j-simple "1.8.0-beta2"]
                 [org.clojure/math.combinatorics "0.1.4"]
                 ]
  :main ^:skip-aot intles.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
