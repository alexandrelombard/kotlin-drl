package com.kdrl.ddpg

import com.kdrl.*
import com.kdrl.dqn.DQN
import com.kdrl.space.Box
import com.kdrl.space.IDiscreteSpace
import com.kdrl.space.ISpace
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

class DDPG<ObservationSpace: ISpace<FloatArray>, ActionSpace: Box<FloatArray>> :
    IDRLTrainer<FloatArray, FloatArray, ObservationSpace, ActionSpace> {

    override val environment: IEnvironment<FloatArray, FloatArray, ObservationSpace, ActionSpace>

//    private val actorModel: MultiLayerNetwork
//    private val targetActorModel: MultiLayerNetwork
//
//    private val criticModel: MultiLayerNetwork
//    private val targetCriticModel: MultiLayerNetwork
//
//    private val replayMemory: MemoryBuffer<FloatArray, FloatArray>
//
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
//        this.replayMemory = MemoryBuffer(replayMemorySize)
//
//        this.model = MultiLayerNetwork(multiLayerConfiguration).wrap()
//        this.model.init()
//        this.targetModel = MultiLayerNetwork(multiLayerConfiguration).wrap()
//        this.targetModel.init()
//        this.targetModel.setParams(this.model.params().dup())
    }

    override fun trainStep(state: FloatArray): Step<FloatArray, FloatArray> {
        TODO("Not yet implemented")
    }
}
