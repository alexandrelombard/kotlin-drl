package com.kdrl.dqn

import com.kdrl.IEnvironment
import com.kdrl.nextStates
import com.kdrl.rewards
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.impl.util.flattenFloats
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.math.max
import kotlin.random.Random

class DQN<ObservationSpace: ISpace<FloatArray>, ActionSpace: IDiscreteSpace>(
    val environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
    val model: InferenceModel,
    val gamma: Float = 0.95f,
    val trainPeriod: Int = 4,
    val updateTargetModelPeriod: Int = 100,
    val batchSize: Int = 1000,
    val replayMemorySize: Int = 10000) {

    val replayMemory = MemoryBuffer<FloatArray, Int>(replayMemorySize)

    var targetModel: InferenceModel
    var stepCount = 0

    init {
        this.targetModel = this.model.copy()

        val model = Sequential.of(
            Input(28,28,1),
            Flatten(),
            Dense(300),
            Dense(100),
            Dense(10)
        )

        model.kGraph().
    }

    fun train(state: FloatArray) {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)
            val futureRewards = mk.ndarray(this.targetModel.predictSoftly(samples.nextStates().toTypedArray().flattenFloats()))

            // Compute updated Q-values
            val updateQValues = mk.ndarray(samples.rewards()) + gamma * futureRewards.

            if(stepCount % updateTargetModelPeriod == 0) {
                this.targetModel = this.model.copy()
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
            model.predict(listOf(state).toTypedArray().flattenFloats())
        }

        return action
    }

    fun epsilonGreedyStrategy(epsilon: Double) {

    }
}
