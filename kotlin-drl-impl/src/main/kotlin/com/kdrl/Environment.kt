package com.kdrl

interface Environment<S: State, A: Action> {
    fun step(action: A): Step<S, A>

    fun sampleAction(): A
}
