package com.kdrl.dqn

import com.kdrl.env.CartPole
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import kotlin.test.Test

class CartPoleDQNTest {
    @Test
    fun testCartPoleDQN() {
        val environment = CartPole()
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(4).nOut(16).build(),
                DenseLayer.Builder().nIn(16).nOut(8).build(),
                OutputLayer.Builder().nIn(8).nOut(environment.actionSpace.size).build()
            )
            .build()
        val dqn = DQN(environment, multiLayerConfiguration)

        dqn.train(1000)
    }
}
