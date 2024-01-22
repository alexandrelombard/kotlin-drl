package com.kdrl

import com.kdrl.space.ISpace

interface IDRLTrainer<State, Action, ObservationSpace: ISpace<State>, ActionSpace: ISpace<Action>> {

    val environment: IEnvironment<State, Action, ObservationSpace, ActionSpace>

    fun train(episode: Int = 1000, afterEpisode: (episodeCount: Int, cumulativeReward: Double) -> Unit = {_, _ -> }) {
        for(i in 0 until episode) {
            var state = environment.reset()
            var done = false
            var cumulativeReward = 0.0

            while (!done) {
                val step = trainStep(state)
                state = step.nextState
                done = step.done
                cumulativeReward += step.reward
            }

            afterEpisode(i, cumulativeReward)
        }
    }

    fun trainStep(state: State): Step<State, Action>
}
