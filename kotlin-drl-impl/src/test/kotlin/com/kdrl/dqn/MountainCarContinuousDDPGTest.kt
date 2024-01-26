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
import kotlin.test.Test

class MountainCarContinuousDDPGTest {
    @Test
    fun testMountainCarContinuousDDPG() {
        val innerLayersSize = 128
        val environment = MountainCarContinuous(maxEpisodeLength = 500)
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(4).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(),
                OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(innerLayersSize).nOut(environment.actionSpace.shape[0]).activation(
                    Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build()
            )
            .backpropType(BackpropType.Standard)
            .build()

        val ddpg = DDPG(environment, actorConfiguration = multiLayerConfiguration,
            criticConfiguration = multiLayerConfiguration,
            updateTargetModel = TARGET_UPDATE_BY_POLYAK_AVERAGING(0.9))
    }
}
