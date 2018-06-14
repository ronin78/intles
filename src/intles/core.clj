(ns intles.core
  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           (org.deeplearning4j.eval Evaluation)
           (org.deeplearning4j.nn.conf MultiLayerConfiguration)
           (org.deeplearning4j.nn.conf NeuralNetConfiguration)
           (org.deeplearning4j.nn.conf.layers DenseLayer)
           (org.deeplearning4j.nn.conf.layers OutputLayer)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.nd4j.linalg.activations Activation)
           (org.nd4j.linalg.api.ndarray INDArray)
           (org.nd4j.linalg.dataset DataSet)
           (org.nd4j.linalg.dataset.api.iterator SamplingDataSetIterator)
           (org.nd4j.linalg.learning.config Nesterovs)
           (org.nd4j.linalg.lossfunctions LossFunctions)
           (org.slf4j Logger)
           (org.slf4j LoggerFactory)
           (org.nd4j.linalg.dataset DataSet)
           (org.nd4j.linalg.api.ndarray INDArray)
           (org.nd4j.linalg.factory Nd4j)
           )
  (:require [clojure.math.combinatorics :as combo])
  )

;; Functions and variables for playing rps
(def veclength 3)
(def feature-size 5)

(defn make-move
  [l]
  (if (empty? l)
    ()
  ( let [f (first l)]
    (if (= f 1)
      (conj (make-move ( rest l)) 0)
      (conj (make-move  (rest l)) 1)
    )))
  )

(defn right?
  [v1 v2]
  (if (zero?  (compare v1 v2))
    1
    0
    )
  )

(defn make-moveset
  [length]
  (combo/selections [0 1] length) 
  )

(def moveset (let [s (make-moveset veclength)]
             (zipmap s (range 0 (count s)))))

(def move-lookup (zipmap (vals moveset) (keys moveset)))

(defn make-moves
  [moveset lookback]
  (partition lookback 1 (cycle (vals moveset)))
  )

(def moves (make-moves moveset (inc feature-size)))

(defn make-feature-vectors
  [moves sample-size]
  (map drop-last (take sample-size moves))
  )

(defn make-label-vector
  [moves sample-size]
  (map last (take sample-size moves))
  )

(defn make-label-vector-wide
  [labels]
  (for [i labels]
    (assoc (vec (repeat 8 0.0)) i 1.0)
    )
  )

(def feature-vec (make-feature-vectors moves 1000))
(def label-vec (make-label-vector-wide ( make-label-vector moves 1000)))

;;Functions and variables for deep learning the game
(def num-rows 100)
(def num-columns 5)
(def output-num 8)
(def batch-size 5)
(def rng-seed 123)
(def num-epochs 100)
(def rate 0.0015)

(def train-set (DataSet. (. Nd4j create (into-array (map double-array feature-vec)))
                        (. Nd4j create (into-array  (map double-array label-vec)))))
(def test-set (DataSet. (. Nd4j create (into-array (map double-array feature-vec)))
                        (. Nd4j create (into-array  (map double-array label-vec)))))
(def train-set-iter (SamplingDataSetIterator. train-set 1 (. train-set numExamples)))
(def test-set-iter (SamplingDataSetIterator. test-set 1 (. test-set numExamples)))
(def updater (org.nd4j.linalg.learning.config.Nesterovs. rate 0.98))
(def dl1 (-> (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
            ;(.nIn (*  num-rows num-columns))
            (.nIn num-columns)
            (.nOut (* output-num 3))
            (.build)
            ))
;(def dl2 (-> (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
;            (.nIn 100)
;            (.nOut 24)
;            (.build)
;            ))
(def ol (-> (org.deeplearning4j.nn.conf.layers.OutputLayer$Builder. org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
            (.nIn (* output-num 3))
            (.nOut output-num)
            ;(.nOut 1)
            (.activation Activation/SOFTMAX)
            (.build)
            ))
;;(def mnist-train (MnistDataSetIterator. batch-size true rng-seed))
;;(def mnist-test (MnistDataSetIterator. batch-size false rng-seed))
(def conf (-> (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
              (.seed rng-seed)
              (.activation Activation/RELU)
              (.weightInit WeightInit/XAVIER)
              (.updater updater)
              (.l2 (* rate 0.005))
              (.list)
              (.layer 0 dl1)
;              (.layer 1 dl2)
              (.layer 1 ol)
              (.pretrain false)
              (.backprop true)
              (.build)
              ))
(def mln (MultiLayerNetwork. conf))
(def sil (ScoreIterationListener. 1))
(.init mln)
(. mln setListeners [sil])
(dotimes [n num-epochs] (. mln fit train-set-iter))
;(dotimes [n num-epochs] (. mln fit (. Nd4j create (into-array (map double-array feature-vec))) (int-array label-vec-int)))
;(dotimes [n num-epochs] (. mln fit mnist-train))

(def evaluate (Evaluation. output-num))
(while (. test-set-iter hasNext) 
  (let [next-result (. test-set-iter next)
    output (. mln output (. next-result getFeatureMatrix))]
    (. evaluate eval (. next-result getLabels) output)
    )
  )

(def stats (. evaluate stats))

(defn return-prediction
  [test-seq]
  (seq  (. mln output  (. Nd4j create  (double-array test-seq))))
  )
