(ns dlj.rps
  (:gen-class)
  (:require [clojure.math.combinatorics :as combo])
  )

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

(defn make-moves
  [moveset lookback]
  (partition lookback 1 (cycle (vals moveset)))
  )

(def moves (make-moves moveset feature-size))

(defn make-feature-vectors
  [moves sample-size]
  (map drop-last (take sample-size moves))
  )

(defn make-label-vector
  [moves sample-size]
  (map last (take sample-size moves))
  )
