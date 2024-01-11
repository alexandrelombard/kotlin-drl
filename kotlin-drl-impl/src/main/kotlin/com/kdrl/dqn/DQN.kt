package com.kdrl.dqn

class DQN<State, Action>() {

    val replayMemory = MemoryBuffer<State, Action>()
    val model: Model
    val targetModel: Model

    fun train(state: State, action: Action, reward: Double, newState: State) {

    }

    fun act(): Action {

    }

    fun epsilonGreedyStrategy(epsilon: Double) {

    }
}
