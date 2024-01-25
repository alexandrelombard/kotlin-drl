package com.kdrl.ddpg

import com.kdrl.*
import com.kdrl.dqn.DQN
import com.kdrl.space.Box
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class DDPG<ObservationSpace: ISpace<FloatArray>, ActionSpace: Box<FloatArray>> :
    IDRLTrainer<FloatArray, FloatArray, ObservationSpace, ActionSpace>,
    IDRLPolicy<FloatArray, FloatArray> {

    override val environment: IEnvironment<FloatArray, FloatArray, ObservationSpace, ActionSpace>

    private val actorModel: MultiLayerNetwork
    private val targetActorModel: MultiLayerNetwork

    private val criticModel: MultiLayerNetwork
    private val targetCriticModel: MultiLayerNetwork

    private val replayMemory: MemoryBuffer<FloatArray, FloatArray>

    constructor(
        environment: IEnvironment<FloatArray, FloatArray, ObservationSpace, ActionSpace>,
        actorConfiguration: MultiLayerConfiguration,
        criticConfiguration: MultiLayerConfiguration,
        gamma: Float = 0.99f,
        trainPeriod: Int = 1,
        updateTargetModel: (dqn: DQN<*, *>) -> Unit = DQN.TARGET_UPDATE_BY_COPY(4),
        batchSize: Int = 128,
        doubleDqn: Boolean = true,
        replayMemorySize: Int = 10000
    ) {
        this.environment = environment
//        this.gamma = gamma
//        this.trainPeriod = trainPeriod
//        this.updateTargetModel = updateTargetModel
//        this.batchSize = batchSize
//        this.doubleDqn = doubleDqn
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
        TODO()
    }

    override fun act(observation: FloatArray): FloatArray {
        // Get actions from actor model
        val sampledActions = this.actorModel.output(Nd4j.create(observation))

        // Add some noise
//        val noise = noise()
        TODO()
    }

//    private fun noise(): INDArray {
//        Nd4j.n
//    }
}
