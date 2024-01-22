package com.kdrl.dqn

import com.kdrl.env.CartPole
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.test.Test

class CartPoleDQNTest {
    @Test
    fun testCartPoleDQN() {
        val environment = CartPole()
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(4).nOut(32).activation(Activation.SELU).build(),
                DenseLayer.Builder().nIn(32).nOut(32).activation(Activation.SELU).build(),
                OutputLayer.Builder().nIn(32).nOut(environment.actionSpace.size).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build()
            )
            .build()
        val dqn = DQN(environment, multiLayerConfiguration)

        dqn.train(1000)
    }
}
