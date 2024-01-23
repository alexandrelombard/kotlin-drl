package com.kdrl.dqn


import com.kdrl.*
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.ops.NDBase
import kotlin.math.max
import kotlin.random.Random

/**
 * Update a target model by copy: every updateTargetModelPeriod, the model is copied into the target model
 * @param dqn the DQN to update
 * @param updateTargetModelPeriod the update period
 */
fun updateTargetModelByCopy(dqn: DQN<*,*>, updateTargetModelPeriod: Int = 1) {
    if(dqn.stepCount % updateTargetModelPeriod == 0) {
        dqn.targetModel.setParams(dqn.model.params().dup())
    }
}

/**
 * Update a target model by Polyak averaging: every step, the target model parameters p are set to tau * p + (1 - tau) * p'
 * where p' are the model parameters
 * @param dqn the DQN to update
 * @param tau the weight given to the current target model parameters (in the interval [0.0, 1.0]
 */
fun updateTargetModelByPolyakAveraging(dqn: DQN<*,*>, tau: Double = 0.99) {
    dqn.targetModel.setParams(dqn.targetModel.params() * tau + (1.0 - tau) * dqn.model.params())
}

class DQN<ObservationSpace: ISpace<FloatArray>, ActionSpace: IDiscreteSpace>(
    override val environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
    multiLayerConfiguration: MultiLayerConfiguration,
    val gamma: Float = 0.99f,
    val trainPeriod: Int = 1,
    val updateTargetModel: (dqn: DQN<*, *>) -> Unit = TARGET_UPDATE_BY_COPY(1),
    val batchSize: Int = 128,
    replayMemorySize: Int = 10000): IDRLTrainer<FloatArray, Int, ObservationSpace, ActionSpace> {

    val replayMemory = MemoryBuffer<FloatArray, Int>(replayMemorySize)

    var model: MultiLayerNetwork
    var targetModel: MultiLayerNetwork

    var stepCount = 0

    init {
        this.model = MultiLayerNetwork(multiLayerConfiguration)
        this.model.init()

        this.targetModel = MultiLayerNetwork(multiLayerConfiguration)
        this.targetModel.init()
        this.targetModel.setParams(this.model.params().dup())
    }

    override fun trainStep(state: FloatArray): Step<FloatArray, Int> {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)

            val futureRewards = this.targetModel.output(samples.nextStates().toINDArray())
            val rewards = samples.rewards().toINDArray()
            val done = samples.done().toINDArray().castTo(DataType.INT32)
            val notDone = Nd4j.onesLike(done) - done

            // Compute updated Q-values
            val updatedQValues = (rewards + gamma * futureRewards.max(1)).mul(notDone)

            // Create a mask for action that were performed
            val masks = NDBase().oneHot(samples.actions().toTypedArray().toIntArray().toINDArray(), 2, 1, 1.0, 0.0)

            // Fit the model by computing the expected q-values
            val qValues = model.output(samples.states().toINDArray())

            val update = ((Nd4j.onesLike(masks) - masks) * qValues) + masks.mul(updatedQValues.reshape(batchSize.toLong(), 1))

            model.fit(samples.states().toINDArray(), update)

            // Eventually update the target model
            updateTargetModel(this)
        }

        stepCount++

        return step
    }

    var epsilon = 1.0
    var epsilonDecay = 2e-5
    var minEpsilon = 0.05

    fun act(state: FloatArray): Int {
        this.epsilon = max(minEpsilon, epsilon - epsilonDecay)

        val action = if(Random.nextFloat() < this.epsilon) {
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

    companion object {
        // region Target model update strategies
        fun TARGET_UPDATE_BY_COPY(period: Int): (dqn: DQN<*, *>)->Unit = { updateTargetModelByCopy(it, period) }
        fun TARGET_UPDATE_BY_POLYAK_AVERAGING(tau: Double): (dqn: DQN<*, *>)->Unit = { updateTargetModelByPolyakAveraging(it, tau) }
        // endregion
    }
}
