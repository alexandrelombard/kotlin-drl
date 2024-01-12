package com.kdrl.dqn

import com.kdrl.IEnvironment
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import kotlin.math.max
import kotlin.random.Random

class DQN<Observation, ObservationSpace: ISpace<Observation>, ActionSpace: IDiscreteSpace>(
    val environment: IEnvironment<Observation, Int, ObservationSpace, ActionSpace>,
    val model: InferenceModel,
    val trainPeriod: Int = 4,
    val updateTargetModelPeriod: Int = 100,
    val batchSize: Int = 1000,
    val replayMemorySize: Int = 10000) {

    val replayMemory = MemoryBuffer<ObservationSpace, ActionSpace>(replayMemorySize)

    var targetModel: InferenceModel
    var stepCount = 0

    init {
        this.targetModel = this.model.copy()
    }

    fun train(state: ObservationSpace, action: ActionSpace, reward: Double, newState: ObservationSpace) {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)
            val futureRewards = this.targetModel.predict(samples.map { it.nextState })

            // Compute updated Q-values

            if(stepCount % updateTargetModelPeriod == 0) {
                this.targetModel = this.model.copy()
            }
        }

        stepCount++
    }

    var epsilon = 1.0
    var epsilonDecay = 0.001
    var minEpsilon = 0.1

    fun act(state: ObservationSpace): Int {
        val value = max(minEpsilon, epsilon - epsilonDecay * stepCount)

        val action = if(Random.nextFloat() < epsilon) {
            // Random action
            environment.actionSpace.sample()
        } else {
            // Action from model
            model.predict(listOf(state))
        }

        return action
    }

    fun epsilonGreedyStrategy(epsilon: Double) {

    }
}
