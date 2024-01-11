package com.kdrl

interface Environment<State, Action> {
    fun step(action: Action): Step<State, Action>

    fun List<State>.flatten(): FloatArray
}
