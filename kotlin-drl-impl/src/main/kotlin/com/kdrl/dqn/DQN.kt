package com.kdrl.dqn


import com.kdrl.*
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
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

    fun train(episode: Int = 1000) {
        for(i in 0 until episode) {
            println("Episode #${i} starting...")
            var state = environment.reset()
            var done = false

            while (!done) {
                val step = trainStep(state)
                done = step.done
            }

            println("Episode #${i} done")
        }
    }

    fun trainStep(state: FloatArray): Step<FloatArray, Int> {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)
            val futureRewards = this.targetModel.output(samples.nextStates().toINDArray())

            // Compute updated Q-values
            val updatedQValues = samples.rewards().toINDArray() + gamma * futureRewards.max(1)
            val masks = samples.actions().toINDArray()

            // Fit the model
            val qValues = model.output(samples.states().toINDArray())
            val qAction = qValues.mul(masks).sum(1)
            model.fit(updatedQValues, qAction)

            // Eventually update the target model
            if(stepCount % updateTargetModelPeriod == 0) {
                this.targetModel.setParams(this.model.params().dup())
            }
        }

        stepCount++

        return step
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
            val input = listOf(state).toINDArray()
            val output = model.output(input)

            output.argMax().getInt(0)
        }

        return action
    }
}
