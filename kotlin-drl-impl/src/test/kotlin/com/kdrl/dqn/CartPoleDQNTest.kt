package com.kdrl.dqn

import com.kdrl.env.CartPole
import org.apache.commons.math3.geometry.spherical.twod.Vertex
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.impl.loss.HuberLoss
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.text.NumberFormat
import java.util.*
import kotlin.test.Test

class CartPoleDQNTest {

    private fun resultFormater(episodeCount: Int, cumulativeReward: Double) {
        println("$episodeCount\t${NumberFormat.getNumberInstance(Locale.FRANCE).format(cumulativeReward)}")
    }

    @Test
    fun testCartPoleDQN() {
        // Checked, seems OK
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
        val dqn = DQN(environment, multiLayerConfiguration, doubleDqn = false)

        dqn.train(1000, this::resultFormater)
    }

    @Test
    fun testCartPoleDoubleDQN() {
        // Checked, seems OK
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
        val dqn = DQN(environment, multiLayerConfiguration, doubleDqn = true,
            updateTargetModel = DQN.TARGET_UPDATE_BY_POLYAK_AVERAGING(0.9))

        dqn.train(1000, this::resultFormater)
    }

    @Test
    fun testCartPoleDuelingDQN() {
        val innerLayersSize = 128
        val environment = CartPole(maxEpisodeLength = 500)
        val neuralNetConfiguration = NeuralNetConfiguration.Builder()
            .updater(Adam(1e-4))
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .graphBuilder()
            .addInputs("Input")
            .addLayer("F1", DenseLayer.Builder().nIn(4).nOut(innerLayersSize).activation(Activation.RELU).build(), "Input")
            .addLayer("F2", DenseLayer.Builder().nIn(innerLayersSize).nOut(innerLayersSize).activation(Activation.RELU).build(), "F1")
            .addLayer("V1", DenseLayer.Builder().nIn(innerLayersSize).nOut(1).activation(Activation.IDENTITY).build(), "F2")
            .addLayer("A1", DenseLayer.Builder().nIn(innerLayersSize).nOut(environment.actionSpace.size).activation(Activation.IDENTITY).build(), "F2")
            .addVertex("AAvg", ElementWiseVertex(ElementWiseVertex.Op.Average), "A1")
            .addVertex("A2", ElementWiseVertex(ElementWiseVertex.Op.Subtract), "A1", "AAvg")
            .addVertex("Q", ElementWiseVertex(ElementWiseVertex.Op.Add), "V1", "A2")
            .setOutputs("Q")
            .build()

        val network = ComputationGraph(neuralNetConfiguration)
        val dqn = DQN(environment, neuralNetConfiguration, doubleDqn = false)

        dqn.train(1000, this::resultFormater)
    }
}
