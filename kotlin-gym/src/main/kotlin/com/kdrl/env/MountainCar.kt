package com.kdrl.env

import com.kdrl.IEnvironment
import com.kdrl.Step
import com.kdrl.space.Box
import com.kdrl.space.Discrete
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.random.Random

class MountainCar(
    val force: Float = 0.001f,
    val gravity: Float = 0.0025f,
    val minPosition: Float = -1.2f,
    val maxPosition: Float = 0.6f,
    val maxSpeed: Float = 0.07f,
    val goalPosition: Float = 0.5f,
    val goalVelocity: Float = 0.0f,
    val maxEpisodeLength: Int = 500,
    private val random: Random = Random.Default) : IEnvironment<FloatArray, Int, Box<FloatArray>, Discrete> {
    val description = "Reproduction in Kotlin of the environment MountainCar-v0"

    override val observationSpace =
        Box<FloatArray>(
            floatArrayOf(-1.2f, -maxSpeed),
            floatArrayOf(0.6f, maxSpeed),
            intArrayOf(2)
        )
    override val actionSpace = Discrete(3)

    var episodeLength = 0
    var state = InternalState()

    override fun reset(): FloatArray {
        episodeLength = 0
        InternalState().let {
            state = it
            return it.toObservation()
        }
    }

    override fun step(action: Int): Step<FloatArray, Int> {
        var done = false

        // Update the state according to the action
        val state = this.state  // Save a copy of the current state
        val newVelocity = (state.velocity + (action - 1) * force + cos(3 * state.position) * (-gravity)).coerceIn(-maxSpeed, maxSpeed)
        val newPosition = (state.position + newVelocity).coerceIn(minPosition, maxPosition)
        val nextState = InternalState(
            position = newPosition,
            velocity = newVelocity
        )

        // Check termination conditions
        val failure = episodeLength > maxEpisodeLength
        if(failure || (state.position >= goalPosition && state.velocity >= goalVelocity)) {
            done = true
        }

        // Increase episode length and update state
        episodeLength += 1
        this.state = nextState

        // Compute the reward
        val reward = -1.0f

        return Step(state.toObservation(), action, nextState.toObservation(), reward, done)
    }

    inner class InternalState(val position: Float = (random.nextFloat() * 0.2f) - 0.6f, val velocity: Float = 0f) {
        fun toObservation(): FloatArray {
            return floatArrayOf(position, velocity)
        }
    }
}
