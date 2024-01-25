package com.kdrl

class MemoryBuffer<State, Action>(val maxSize: Int): IMemoryBuffer<State, Action> {

    private val buffer = arrayListOf<Step<State, Action>>()

    override val size: Int
        get() = this.buffer.size

    override fun push(step: Step<State, Action>) {
        buffer.add(step)

        if(buffer.size >= maxSize) {
            buffer.removeFirst()
        }
    }

    override fun sample(sampleSize: Int): List<Step<State, Action>> {
        if(sampleSize < 0) {
            throw IllegalArgumentException("Sample size must be positive")
        }

        return buffer.shuffled().slice(0..<sampleSize)
    }
}
