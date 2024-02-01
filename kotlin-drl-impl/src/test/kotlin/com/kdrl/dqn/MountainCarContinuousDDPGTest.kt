package com.kdrl.dqn

import com.kdrl.ddpg.DDPG
import com.kdrl.ddpg.DDPG.Companion.TARGET_UPDATE_BY_POLYAK_AVERAGING
import com.kdrl.env.CartPole
import com.kdrl.env.MountainCarContinuous
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.text.NumberFormat
import java.util.*
import kotlin.test.Test

class MountainCarContinuousDDPGTest {

    private fun resultFormater(episodeCount: Int, cumulativeReward: Double) {
        println("$episodeCount\t${NumberFormat.getNumberInstance(Locale.FRANCE).format(cumulativeReward)}")
    }

    @Test
    fun testMountainCarContinuousDDPG() {
        val innerLayersSize = 128
        val environment = MountainCarContinuous(maxEpisodeLength = 500)
        val actorConfiguration = NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(2).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(environment.actionSpace.shape[0]).activation(Activation.IDENTITY).build()
            )
            .backpropType(BackpropType.Standard)
            .build()

        val criticConfiguration = NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(2 + 1).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(),
                OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(innerLayersSize).nOut(1).activation(Activation.IDENTITY).build()
            )
            .backpropType(BackpropType.Standard)
            .build()

        val ddpg = DDPG(environment, actorConfiguration = actorConfiguration,
            criticConfiguration = criticConfiguration,
            updateTargetModel = TARGET_UPDATE_BY_POLYAK_AVERAGING(0.9))

        ddpg.train(1000, this::resultFormater)
    }
}
