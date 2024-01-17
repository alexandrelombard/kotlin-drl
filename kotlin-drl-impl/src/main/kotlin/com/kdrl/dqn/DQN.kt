package com.kdrl.dqn


import com.kdrl.IEnvironment
import com.kdrl.nextStates
import com.kdrl.rewards
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.max
import kotlin.random.Random

class DQN<ObservationSpace: ISpace<FloatArray>, ActionSpace: IDiscreteSpace>(
    val environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
    multiLayerConfiguration: MultiLayerConfiguration,
    val gamma: Float = 0.95f,
    val trainPeriod: Int = 4,
    val updateTargetModelPeriod: Int = 100,
    val batchSize: Int = 1000,
    val replayMemorySize: Int = 10000) {

    val replayMemory = MemoryBuffer<FloatArray, Int>(replayMemorySize)

    var model: MultiLayerNetwork
    var targetModel: MultiLayerNetwork

    var stepCount = 0

    init {
        this.model = MultiLayerNetwork(multiLayerConfiguration)
        this.model.init()

        this.targetModel = MultiLayerNetwork(multiLayerConfiguration)
        this.targetModel.setParams(this.model.params().dup())
        this.targetModel.init() // FIXME Check if this should be done before params copy
    }

    fun train(state: FloatArray) {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)
            val futureRewards = this.targetModel.output(samples.nextStates().toTypedArray().flattenFloats())

            // Compute updated Q-values
            val updateQValues = mk.ndarray(samples.rewards()) + gamma * futureRewards

            if(stepCount % updateTargetModelPeriod == 0) {
                this.targetModel.setParams(this.model.params().dup())
            }
        }

        stepCount++
    }

    var epsilon = 1.0
    var epsilonDecay = 0.001
    var minEpsilon = 0.1

    fun act(state: FloatArray): Int {
        val epsilonValue = max(minEpsilon, epsilon - epsilonDecay * stepCount)

        val action = if(Random.nextFloat() < epsilonValue) {
            // Random action
            environment.actionSpace.sample()
        } else {
            // Action from model
            val features = Nd4j.create(listOf(state).toTypedArray())
            val labels = Nd4j.create(0)
            val dataSetIterator = INDArrayDataSetIterator(listOf(org.nd4j.common.primitives.Pair(features, labels)), 1)
            model.evaluate(dataSetIterator)
            model.predict(listOf(state).toTypedArray().flattenFloats())
        }

        return action
    }

    fun epsilonGreedyStrategy(epsilon: Double) {

    }
}
