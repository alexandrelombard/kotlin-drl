package com.kdrl

interface IMemoryBuffer<State, Action> {
    val size: Int

    fun push(step: Step<State, Action>)

    fun sample(sampleSize: Int): List<Step<State, Action>>
}
