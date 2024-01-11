package com.kdrl.dqn

import com.kdrl.Step

class MemoryBuffer<State, Action>(val size: Int) {

    val buffer = arrayListOf<Step<State, Action>>()

    fun push(step: Step<State, Action>) {
        buffer.add(step)

        if(buffer.size >= size) {
            buffer.removeFirst()
        }
    }

    fun sample(sampleSize: Int): List<Step<State, Action>> {
        if(sampleSize < 0) {
            throw IllegalArgumentException("Sample size must be positive")
        }

        return buffer.shuffled().slice(0..<sampleSize)
    }
}
