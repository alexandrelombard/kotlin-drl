package com.kdrl

data class Step<State, Action>(
    val state: State,
    val action: Action,
    val nextState: State,
    val reward: Double,
    val done: Boolean)

fun List<Step<*, *>>.rewards(): DoubleArray {
    return this.map { it.reward }.toTypedArray().toDoubleArray()
}

fun <State> List<Step<State, *>>.states(): List<State> {
    return this.map { it.state }.toList()
}

fun <State> List<Step<State, *>>.nextStates(): List<State> {
    return this.map { it.nextState }.toList()
}

fun List<Step<*, *>>.done(): BooleanArray {
    return this.map { it.done }.toBooleanArray()
}

fun <Action> List<Step<*, Action>>.actions(): List<Action> {
    return this.map { it.action }.toList()
}
