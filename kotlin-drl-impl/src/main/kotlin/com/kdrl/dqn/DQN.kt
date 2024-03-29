package com.kdrl.dqn


import com.kdrl.*
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.ops.NDBase
import kotlin.math.max
import kotlin.random.Random

/**
 * Update a target model by copy: every updateTargetModelPeriod, the model is copied into the target model
 * @param dqn the DQN to update
 * @param updateTargetModelPeriod the update period
 */
fun updateTargetModelByCopy(dqn: DQN<*,*>, updateTargetModelPeriod: Int = 4) {
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

class DQN<ObservationSpace: ISpace<FloatArray>, ActionSpace: IDiscreteSpace> :
    IDRLTrainer<FloatArray, Int, ObservationSpace, ActionSpace>,
    IDRLPolicy<FloatArray, Int> {
    override val environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>

    val gamma: Float
    val trainPeriod: Int
    val updateTargetModel: (dqn: DQN<*, *>) -> Unit
    val batchSize: Int
    val doubleDqn: Boolean

    constructor(
        environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
        multiLayerConfiguration: MultiLayerConfiguration,
        gamma: Float = 0.99f,
        trainPeriod: Int = 1,
        updateTargetModel: (dqn: DQN<*, *>) -> Unit = TARGET_UPDATE_BY_COPY(4),
        batchSize: Int = 128,
        doubleDqn: Boolean = true,
        replayMemorySize: Int = 10000
    ) {
        this.environment = environment
        this.gamma = gamma
        this.trainPeriod = trainPeriod
        this.updateTargetModel = updateTargetModel
        this.batchSize = batchSize
        this.doubleDqn = doubleDqn
        this.replayMemory = MemoryBuffer(replayMemorySize)

        this.model = MultiLayerNetwork(multiLayerConfiguration).wrap()
        this.model.init()
        this.targetModel = MultiLayerNetwork(multiLayerConfiguration).wrap()
        this.targetModel.init()
        this.targetModel.setParams(this.model.params().dup())
    }

    constructor(
        environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
        computationGraphConfiguration: ComputationGraphConfiguration,
        gamma: Float = 0.99f,
        trainPeriod: Int = 1,
        updateTargetModel: (dqn: DQN<*, *>) -> Unit = TARGET_UPDATE_BY_COPY(4),
        batchSize: Int = 128,
        doubleDqn: Boolean = true,
        replayMemorySize: Int = 10000
    ) {
        this.environment = environment
        this.gamma = gamma
        this.trainPeriod = trainPeriod
        this.updateTargetModel = updateTargetModel
        this.batchSize = batchSize
        this.doubleDqn = doubleDqn
        this.replayMemory = MemoryBuffer(replayMemorySize)

        this.model = ComputationGraph(computationGraphConfiguration).wrap()
        this.model.init()
        this.targetModel = ComputationGraph(computationGraphConfiguration).wrap()
        this.targetModel.init()
        this.targetModel.setParams(this.model.params().dup())
    }

    val replayMemory: MemoryBuffer<FloatArray, Int>

    val model: NeuralNetworkWrapper<*>
    val targetModel: NeuralNetworkWrapper<*>

    var stepCount = 0

    override fun trainStep(state: FloatArray): Step<FloatArray, Int> {
        val action = this.act(state)
        val step = environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            val samples = this.replayMemory.sample(batchSize)

            val states = samples.states().toINDArray()
            val nextStates = samples.nextStates().toINDArray()
            val actions = samples.actions().toTypedArray().toIntArray().toINDArray()
            val rewards = samples.rewards().toINDArray()
            val done = samples.done().toINDArray().castTo(DataType.INT32)
            val notDone = Nd4j.onesLike(done) - done

            // Create a mask for action that were performed
            val masks = NDBase().oneHot(actions, this.environment.actionSpace.size, 1, 1.0, 0.0)

            // Get the current Q-values from the model
            val qValues = this.model.output(states)

            // Estimate future rewards using target model
            val update: INDArray

            if(doubleDqn) {
                val futureRewards = this.model.output(nextStates)
                val futureActions = this.targetModel.output(nextStates).argMax(1)
                val futureActionsMask = NDBase().oneHot(futureActions, 2, 1, 1.0, 0.0)

                // Compute updated Q-values
                val updatedQValues = (rewards + gamma * (futureRewards * futureActionsMask).sum(1)).mul(notDone)

                // Fit the model by computing the expected q-values
                update = ((Nd4j.onesLike(masks) - masks) * qValues) + masks.mul(updatedQValues.reshape(batchSize.toLong(), 1))
            } else {
                val futureRewards = this.targetModel.output(nextStates)

                // Compute updated Q-values
                val updatedQValues = (rewards + gamma * futureRewards.max(1)).mul(notDone)

                // Fit the model by computing the expected q-values
                update = ((Nd4j.onesLike(masks) - masks) * qValues) + masks.mul(updatedQValues.reshape(batchSize.toLong(), 1))

            }

            this.model.fit(states, update)

            // Eventually update the target model
            updateTargetModel(this)
        }

        stepCount++

        return step
    }

    var epsilon = 1.0
    var epsilonDecay = 2e-5
    var minEpsilon = 0.05

    override fun act(observation: FloatArray): Int {
        this.epsilon = max(minEpsilon, epsilon - epsilonDecay)

        val action = if(Random.nextFloat() < this.epsilon) {
            // Random action
            environment.actionSpace.sample()
        } else {
            // Action from model
            val input = listOf(observation).toINDArray()
            val output = model.output(input)

            output.argMax().getInt(0)
        }

        return action
    }

    companion object {
        // region Target model update strategies
        /**
         * @param updateTargetModelPeriod the update period
         */
        fun TARGET_UPDATE_BY_COPY(period: Int): (dqn: DQN<*, *>)->Unit = { updateTargetModelByCopy(it, period) }

        /**
         * @param tau the weight given to the current target model parameters (in the interval [0.0, 1.0]
         */
        fun TARGET_UPDATE_BY_POLYAK_AVERAGING(tau: Double): (dqn: DQN<*, *>)->Unit = { updateTargetModelByPolyakAveraging(it, tau) }
        // endregion
    }
}
