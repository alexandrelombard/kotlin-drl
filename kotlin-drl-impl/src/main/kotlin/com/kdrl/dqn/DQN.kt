package com.kdrl.dqn

import com.kdrl.DiscreteAction
import com.kdrl.Environment
import com.kdrl.State
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import kotlin.math.max
import kotlin.random.Random

class DQN<S: State, A: DiscreteAction>(
    val environment: Environment<S, A>,
    val model: InferenceModel,
    val trainPeriod: Int = 4,
    val updateTargetModelPeriod: Int = 100,
    val batchSize: Int = 1000,
    val replayMemorySize: Int = 10000) {

    val replayMemory = MemoryBuffer<S, A>(replayMemorySize)

    var targetModel: InferenceModel
    var stepCount = 0

    init {
        this.targetModel = this.model.copy()
    }

    fun train(state: S, action: A, reward: Double, newState: S) {
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

    fun act(state: S): A {
        val value = max(minEpsilon, epsilon - epsilonDecay * stepCount)

        val action = if(Random.nextFloat() < epsilon) {
            // Random action
            environment.sampleAction()
        } else {
            // Action from model
            return model.predict(listOf(state))
        }
    }

    fun epsilonGreedyStrategy(epsilon: Double) {

    }
}
