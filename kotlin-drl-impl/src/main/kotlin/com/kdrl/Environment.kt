package com.kdrl

interface Environment<State, A: Action> {
    fun step(action: A): Step<State, A>

    fun sampleAction(): A

    fun List<State>.flatten(): FloatArray
}
