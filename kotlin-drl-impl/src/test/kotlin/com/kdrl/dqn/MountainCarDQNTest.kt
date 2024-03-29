package com.kdrl.dqn

import com.kdrl.env.MountainCar
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

class MountainCarDQNTest {
    private fun resultFormater(episodeCount: Int, cumulativeReward: Double) {
        println("$episodeCount\t${NumberFormat.getNumberInstance(Locale.FRANCE).format(cumulativeReward)}")
    }
    /**
     * This function tests the DQN on the MountainCar environment
     */
    @Test
    fun testMountainCarDQN() {
        // Checked, seems OK
        val innerLayersSize = 128
        val environment = MountainCar(maxEpisodeLength = 500)
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(
                DenseLayer.Builder().nIn(2).nOut(innerLayersSize).activation(Activation.RELU).build(),
                DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(),
                OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(innerLayersSize).nOut(environment.actionSpace.size).activation(Activation.IDENTITY).build()
            )
            .backpropType(BackpropType.Standard)
            .build()
        val dqn = DQN(environment, multiLayerConfiguration, doubleDqn = false)

        dqn.train(1000, this::resultFormater)
    }
}
