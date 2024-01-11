package com.kdrl.dqn

class MemoryBuffer<State, Action>(val size: Double) {
    typealias Step = (State, Action, State, Double, Boolean)

    val buffer: List<Step>

    fun push(step: Step) {
        buffer.add(step)
    }
}
