package com.kdrl.dqn

import com.kdrl.env.CartPole
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.impl.loss.HuberLoss
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.test.Test

class CartPoleDQNTest {
    @Test
    fun testCartPoleDQN() {
        val innerLayersSize = 128
        val environment = CartPole(maxEpisodeLength = 500)
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(4).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(),
                OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(innerLayersSize).nOut(environment.actionSpace.size).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build()
            )
            .backpropType(BackpropType.Standard)
            .build()
        val dqn = DQN(environment, multiLayerConfiguration)

        dqn.train(1000) { episodeCount, cumulativeReward ->
            println("Episode #$episodeCount: $cumulativeReward")
        }
    }
}
