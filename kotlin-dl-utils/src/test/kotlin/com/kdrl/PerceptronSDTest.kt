package com.kdrl

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer
import org.deeplearning4j.nn.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.weightinit.impl.XavierInitScheme
import kotlin.random.Random
import kotlin.test.Test

class PerceptronSDTest {
    @Test
    fun testPerceptronWithSameDiff() {
//        // Building the neural net
//        val conf = NeuralNetConfiguration.Builder()
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .updater(Adam(0.01))
//            .list(
//                SameDiffLambdaLayer.Builder()
//                SameDiffLayer(DenseLayer.Builder().nIn(2).nOut(3).activation(Activation.IDENTITY).build().conf),
//                DenseLayer.Builder().nIn(2).nOut(3).activation(Activation.IDENTITY).build(),
//                DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.IDENTITY).build(),
//                OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(3).nOut(1).build())
//            .backpropType(BackpropType.Standard)
//            .build()
//
//        val neuralNet = MultiLayerNetwork(conf)
//        neuralNet.init()
//
//        // Building the XOR dataset
//        val trainingSetSize = 100000
//        val features = Nd4j.create(trainingSetSize, 2)
//        val labels = Nd4j.create(trainingSetSize, 1)
//
//        for(i in 0 until trainingSetSize) {
//            val f1 = Random.nextFloat() * 100
//            val f2 = Random.nextFloat() * 100
//            val l = f1 * 5 + f2
//
//            features.putScalar(intArrayOf(i, 0), f1)
//            features.putScalar(intArrayOf(i, 1), f2)
//            labels.putScalar(intArrayOf(i, 0), l)
//        }
//
//        // Train
//        sd.fit()
    }
}
