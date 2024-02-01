package com.kdrl.ddpg

import com.kdrl.*
import com.kdrl.space.Box
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.rng.distribution.Distribution
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import kotlin.math.exp

/**
 * Update a target model by copy: every updateTargetModelPeriod, the model is copied into the target model
 * @param dqn the DDPG to update
 * @param updateTargetModelPeriod the update period
 */
fun updateTargetModelByCopy(ddpg: DDPG<*,*>, destinationModel: MultiLayerNetwork, sourceModel: MultiLayerNetwork, updateTargetModelPeriod: Int = 4) {
    if(ddpg.stepCount % updateTargetModelPeriod == 0) {
        destinationModel.setParams(sourceModel.params().dup())
    }
}

/**
 * Update a target model by Polyak averaging: every step, the target model parameters p are set to tau * p + (1 - tau) * p'
 * where p' are the model parameters
 * @param tau the weight given to the current target model parameters (in the interval [0.0, 1.0]
 */
fun updateTargetModelByPolyakAveraging(destinationModel: MultiLayerNetwork, sourceModel: MultiLayerNetwork, tau: Double = 0.99) {
    destinationModel.setParams(destinationModel.params() * tau + (1.0 - tau) * sourceModel.params())
}

class DDPG<ObservationSpace: ISpace<FloatArray>, ActionSpace: Box<FloatArray>> :
    IDRLTrainer<FloatArray, FloatArray, ObservationSpace, ActionSpace>,
    IDRLPolicy<FloatArray, FloatArray> {

    override val environment: IEnvironment<FloatArray, FloatArray, ObservationSpace, ActionSpace>

    private val actorModel: MultiLayerNetwork
    private val targetActorModel: MultiLayerNetwork

    private val criticModel: MultiLayerNetwork
    private val targetCriticModel: MultiLayerNetwork

    val updateTargetModel: (ddpg: DDPG<*, *>) -> Unit

    val noiseDistribution: Distribution
    val batchSize: Int
    val gamma: Float
    val trainPeriod: Int
    private val replayMemory: MemoryBuffer<FloatArray, FloatArray>

    var stepCount = 0

    constructor(
        environment: IEnvironment<FloatArray, FloatArray, ObservationSpace, ActionSpace>,
        actorConfiguration: MultiLayerConfiguration,
        criticConfiguration: MultiLayerConfiguration,
        gamma: Float = 0.99f,
        trainPeriod: Int = 1,
        updateTargetModel: (ddpg: DDPG<*, *>) -> Unit = TARGET_UPDATE_BY_POLYAK_AVERAGING(0.9),
        batchSize: Int = 128,
        noiseDistribution: Distribution = NormalDistribution(0.0, 0.1),
        replayMemorySize: Int = 10000
    ) {
        this.environment = environment
        this.gamma = gamma
        this.trainPeriod = trainPeriod
        this.updateTargetModel = updateTargetModel
        this.batchSize = batchSize
        this.noiseDistribution = noiseDistribution
        this.replayMemory = MemoryBuffer(replayMemorySize)

        // Initialize the models
        this.actorModel = MultiLayerNetwork(actorConfiguration)
        this.actorModel.init()
        this.targetActorModel = MultiLayerNetwork(actorConfiguration)
        this.targetActorModel.init()
        this.targetActorModel.setParams(this.actorModel.params().dup())

        this.criticModel = MultiLayerNetwork(criticConfiguration)
        this.criticModel.init()
        this.targetCriticModel = MultiLayerNetwork(criticConfiguration)
        this.targetCriticModel.init()
        this.targetCriticModel.setParams(this.criticModel.params().dup())
    }

    override fun trainStep(state: FloatArray): Step<FloatArray, FloatArray> {
        // Run action, store the step in memory
        val action = act(state)
        val step = this.environment.step(action)

        this.replayMemory.push(step)

        if(stepCount % trainPeriod == 0 && this.replayMemory.size > batchSize) {
            // Fetch the training batch
            val samples = this.replayMemory.sample(this.batchSize)
            val rewards = samples.rewards().toINDArray()
            val states = samples.states().toINDArray()
            val nextStates = samples.nextStates().toINDArray()
            val actions = samples.actions().toINDArray()

            // Perform training for critic
            run {
                val targetActions = this.targetActorModel.output(nextStates, true)
                val expectedCriticValue =
                    rewards.reshape(-1, 1) + this.gamma * this.targetCriticModel.output(Nd4j.hstack(nextStates, targetActions), true)
                val criticValue = this.criticModel.output(Nd4j.hstack(states, actions), true)
//                val criticLoss = Transforms.pow(criticValue - expectedCriticValue, 2).mean()

                this.criticModel.fit(Nd4j.hstack(states, actions), expectedCriticValue)
//                this.criticModel.updateWithExternalError(Nd4j.hstack(states, actions), criticLoss)
            }

            // Perform training for actor
            run {
                val modelActions = this.actorModel.output(states, true)
                val criticValue = this.criticModel.output(Nd4j.hstack(states, modelActions), true)
                val actorLoss = -criticValue.mean()

//                this.actorModel.fit(states, modelActions + actorLoss)
                this.actorModel.updateWithExternalError(states, actorLoss)
            }

            updateTargetModel(this)
        }

        // Update the step count
        this.stepCount++

        return step
    }

    override fun act(observation: FloatArray): FloatArray {
        // Get actions from actor model
        val sampledAction = this.actorModel.output(Nd4j.create(observation).reshape(1, -1))

        // Add some noise
        val noise = Nd4j.rand(noiseDistribution, *sampledAction.shape().toTypedArray().toLongArray())
        val actionWithNoise = sampledAction + noise

        return actionWithNoise.toFloatVector()
    }

    companion object {
        // region Target model update strategies
        /**
         * @param updateTargetModelPeriod the update period
         */
        fun TARGET_UPDATE_BY_COPY(period: Int): (ddpg: DDPG<*, *>)->Unit = {
            updateTargetModelByCopy(it, it.targetActorModel, it.actorModel, period)
            updateTargetModelByCopy(it, it.targetCriticModel, it.criticModel, period)
        }

        /**
         * @param tau the weight given to the current target model parameters (in the interval [0.0, 1.0]
         */
        fun TARGET_UPDATE_BY_POLYAK_AVERAGING(tau: Double): (ddpg: DDPG<*, *>)->Unit = {
            updateTargetModelByPolyakAveraging(it.targetActorModel, it.actorModel, tau)
            updateTargetModelByPolyakAveraging(it.targetCriticModel, it.criticModel, tau)
        }
        // endregion
    }
}
