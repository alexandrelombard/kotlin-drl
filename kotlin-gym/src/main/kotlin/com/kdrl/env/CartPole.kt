package com.kdrl.env

import com.kdrl.IEnvironment
import com.kdrl.Step
import com.kdrl.space.Box
import com.kdrl.space.Discrete
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

class CartPole(
    val gravity: Float = 9.8f,
    val massCart: Float = 1.0f,
    val massPole: Float = 1.0f,
    val length: Float = 0.5f,
    val forceMag: Float = 10.0f,
    val tau: Float = 0.02f) : IEnvironment<FloatArray, Int, Box<FloatArray>, Discrete> {

    val totalMass = massCart + massPole
    val poleMassLength = massPole + length

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

        // TODO Check if action is valid

        // Check termination conditions
        if(state.poleAngle < -12 || state.poleAngle > 12 ||
            state.position < -2.4 || state.position > 2.4 ||
            episodeLength > 100) {
            done = true
        }

        // Update the state according to the action
        episodeLength += 1
        val state = this.state

        val force = if(action == 1) forceMag else -forceMag
        val cosTheta = cos(state.poleAngle)
        val sinTheta = sin(state.poleAngle)
        val temp = (force + poleMassLength * state.poleAngularVelocity.pow(2) * sinTheta) / totalMass
        val thetaAcc = (gravity * sinTheta - cosTheta * temp) / (length * (4.0f / 3.0f - massPole * cosTheta.pow(2) / totalMass))
        val xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass

        // Euler update
        val nextState = InternalState(
            state.position + tau * state.velocity,
            state.velocity + tau * xAcc,
            state.poleAngle + tau * state.poleAngularVelocity,
            state.poleAngularVelocity + tau * thetaAcc)

        this.state = nextState

        // Compute the reward
        val reward = 1.0f

        return Step(state.toObservation(), action, nextState.toObservation(), reward, done)
    }

    data class InternalState(var position: Float = 0f, var velocity: Float = 0f, var poleAngle: Float = 0.0f, var poleAngularVelocity: Float = 0.0f) {
        fun toObservation(): FloatArray {
            return floatArrayOf(position, velocity, poleAngle, poleAngularVelocity)
        }
    }
}
