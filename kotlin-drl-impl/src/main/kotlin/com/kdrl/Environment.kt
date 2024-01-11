package com.kdrl

interface Environment<State, Action> {
    fun step(action: Action): (State, Double, Boolean)
}
