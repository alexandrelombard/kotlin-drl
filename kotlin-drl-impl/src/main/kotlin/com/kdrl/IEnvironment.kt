package com.kdrl

import com.kdrl.space.ISpace

interface IEnvironment<Observation, Action, ObservationSpace: ISpace<Observation>, ActionSpace: ISpace<Action>> {
    val observationSpace: ObservationSpace
    val actionSpace: ActionSpace

    /**
     * Performs an action in the environment
     */
    fun step(action: Action): Step<Observation, Action>

    /**
     * Resets the environmnent
     */
    fun reset(): Observation
}
