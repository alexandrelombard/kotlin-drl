package com.kdrl

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.random.Random
import kotlin.test.Test

class PerceptronTest {
    @Test
    fun trainAndEvaluatePerceptron() {
        // Building the neural net
        val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Adam(0.01))
            .list(
                DenseLayer.Builder().nIn(2).nOut(3).activation(Activation.IDENTITY).build(),
                DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.IDENTITY).build(),
                OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(3).nOut(1).build())
            .backpropType(BackpropType.Standard)
            .build()

        val neuralNet = MultiLayerNetwork(conf)
        neuralNet.init()

        // Building the XOR dataset
        val trainingSetSize = 100000
        val features = Nd4j.create(trainingSetSize, 2)
        val labels = Nd4j.create(trainingSetSize, 1)

        for(i in 0 until trainingSetSize) {
//            val f1 = Random.nextInt(0, 2)
//            val f2 = Random.nextInt(0, 2)
//            val l = f1 xor f2
            val f1 = Random.nextFloat() * 100
            val f2 = Random.nextFloat() * 100
            val l = f1 * 5 + f2

            features.putScalar(intArrayOf(i, 0), f1.toFloat())
            features.putScalar(intArrayOf(i, 1), f2.toFloat())
            labels.putScalar(intArrayOf(i, 0), l.toFloat())
        }

        // Train the neural net
        for(i in 0 until 10000) {
            neuralNet.fit(features, labels)
        }

        // Try prediction
//        val testFeature = Nd4j.create(floatArrayOf(1.0f, 0.0f))
//        val testLabel = Nd4j.create(floatArrayOf(1.0f))
//        val dataSetIterator = INDArrayDataSetIterator(listOf(Pair(testFeature, testLabel)), 1)

        for(i in 0 until 100) {
            val testData = Nd4j.zeros(1, 2)
            val x0 = Random.nextFloat() * 100
            val x1 = Random.nextFloat() * 100
            testData.putRow(0, Nd4j.create(floatArrayOf(x0, x1)))
            val output = neuralNet.output(testData)
            println("Output: $output - Expected: ${x0 * 5 + x1}")
        }
    }
}
