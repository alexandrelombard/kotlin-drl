package com.kdrl

data class Step<State, Action>(
    val state: State,
    val action: Action,
    val nextState: State,
    val reward: Double,
    val done: Boolean)
