package com.kdrl.ddqn


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

class DDQN<ObservationSpace: ISpace<FloatArray>, ActionSpace: IDiscreteSpace>(
    override val environment: IEnvironment<FloatArray, Int, ObservationSpace, ActionSpace>,
    multiLayerConfiguration: MultiLayerConfiguration,
    val gamma: Float = 0.99f,
    val trainPeriod: Int = 1,
    val updateTargetModelPeriod: Int = 2,
    val batchSize: Int = 128,
    val replayMemorySize: Int = 10000): IDRLTrainer<FloatArray, Int, ObservationSpace, ActionSpace> {

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
//            val updatedQValues = rewards.mul(done) + (rewards + gamma * futureRewards.max(1)).mul(notDone)
            val updatedQValues = (rewards + gamma * futureRewards.max(1)).mul(notDone)

            // Create a mask for action that were performed
            val masks = NDBase().oneHot(samples.actions().toTypedArray().toIntArray().toINDArray(), 2, 1, 1.0, 0.0)

            // Fit the model by computing the expected q-values
            val qValues = model.output(samples.states().toINDArray())

            // val update = qAction + invertedMasks.mul(updatedQValues.reshape(batchSize.toLong(), 1))
            //val update = qValues + masks.mul(updatedQValues.reshape(batchSize.toLong(), 1))
            val update = ((Nd4j.onesLike(masks) - masks) * qValues) + masks.mul(updatedQValues.reshape(batchSize.toLong(), 1))

            // FIXME
            model.fit(samples.states().toINDArray(), update)
//            println("Estimated loss: ${update.squaredDistance(qValues)}")

            // Eventually update the target model
            if(stepCount % updateTargetModelPeriod == 0) {
                this.targetModel.setParams(this.model.params().dup())
            }
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
}
