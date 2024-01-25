package com.kdrl.env

import com.kdrl.IEnvironment
import com.kdrl.Step
import com.kdrl.space.Box
import kotlin.math.cos
import kotlin.math.pow
import kotlin.random.Random

class MountainCarContinuous(
    val power: Float = 0.0015f,
    val gravity: Float = 0.0025f,
    val minPosition: Float = -1.2f,
    val maxPosition: Float = 0.6f,
    val maxSpeed: Float = 0.07f,
    val goalPosition: Float = 0.5f,
    val goalVelocity: Float = 0.0f,
    private val random: Random = Random.Default) : IEnvironment<FloatArray, FloatArray, Box<FloatArray>, Box<FloatArray>> {
    val description = "Reproduction in Kotlin of the environment MountainCarContinuous-v0"

    override val observationSpace =
        Box<FloatArray>(
            floatArrayOf(-1.2f, -maxSpeed),
            floatArrayOf(0.6f, maxSpeed),
            intArrayOf(2)
        )
    override val actionSpace = Box<FloatArray>(floatArrayOf(-1.0f), floatArrayOf(1.0f), intArrayOf(1))

    var episodeLength = 0
    var state = InternalState()

    override fun reset(): FloatArray {
        episodeLength = 0
        InternalState().let {
            state = it
            return it.toObservation()
        }
    }

    override fun step(action: FloatArray): Step<FloatArray, FloatArray> {
        var done = false

        val force = action[0]

        // Update the state according to the action
        val state = this.state  // Save a copy of the current state
        val newVelocity = (state.velocity + force * power - cos(3 * state.position) * gravity).coerceIn(-maxSpeed, maxSpeed)
        val newPosition = (state.position + newVelocity).coerceIn(minPosition, maxPosition)
        val nextState = InternalState(
            position = newPosition,
            velocity = newVelocity
        )

        // Check termination conditions
        if(state.position >= goalPosition && state.velocity >= goalVelocity) {
            done = true
        }

        // Increase episode length and update state
        episodeLength += 1
        this.state = nextState

        // Compute the reward
        val reward = if(done) 100.0f else -0.1f * force.pow(2)

        return Step(state.toObservation(), action, nextState.toObservation(), reward, done)
    }

    inner class InternalState(val position: Float = (random.nextFloat() * 0.2f) - 0.6f, val velocity: Float = 0f) {
        fun toObservation(): FloatArray {
            return floatArrayOf(position, velocity)
        }
    }
}
