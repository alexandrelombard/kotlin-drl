package com.kdrl.env

import com.kdrl.IEnvironment
import com.kdrl.Step
import com.kdrl.space.Box
import com.kdrl.space.Discrete

class CartPole : IEnvironment<FloatArray, Int, Box<FloatArray>, Discrete> {

    override val observationSpace =
        Box<FloatArray>(
            floatArrayOf(-4.8f, -3.4028235e+38f, -4.1887903e-01f, -3.4028235e+38f),
            floatArrayOf(4.8f, 3.4028235e+38f, 4.1887903e-01f, 3.4028235e+38f),
            intArrayOf(4)
        )
    override val actionSpace = Discrete(2)

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

        // Check termination conditions
        if(state.poleAngle < -12 || state.poleAngle > 12 ||
            state.position < -2.4 || state.position > 2.4 ||
            episodeLength > 100) {
            done = true
        }

        // Update the state according to the action
        episodeLength += 1
        var state = this.state
        var nextState = this.state

        this.state = nextState

        // Compute the reward
        val reward = episodeLength.toFloat()

        return Step(state.toObservation(), action, nextState.toObservation(), reward, done)
    }

    data class InternalState(var position: Float = 0f, var velocity: Float = 0f, var poleAngle: Float = 0.0f, var poleAngularVelocity: Float = 0.0f) {
        fun toObservation(): FloatArray {
            return floatArrayOf(position, velocity, poleAngle, poleAngularVelocity)
        }
    }
}
